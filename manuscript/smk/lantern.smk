from functools import partial
import pickle
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Subset
from dpath.util import get
from tqdm import tqdm
import numpy as np
import pandas as pd
import mlflow
from torch.utils.data import DataLoader

from lantern.dataset import Dataset
from lantern.model import Model
from lantern.model.basis import VariationalBasis
from lantern.model.surface import Phenotype

from src.predict import predictions

def train_lantern(
    input, output, wildcards, tloader, vloader, optimizer, loss, model, lr, epochs
):
    """General purpose lantern training optimization loop
    """

    try:
        with mlflow.start_run() as run:
            mlflow.log_param("dataset", wildcards.ds)
            mlflow.log_param("model", "lantern")
            mlflow.log_param("lr", lr)
            mlflow.log_param("cv", wildcards.cv)
            mlflow.log_param("batch-size", tloader.batch_size)

            pbar = tqdm(range(epochs),)
            for e in pbar:

                # logging of loss values
                tloss = 0
                vloss = 0

                # go through minibatches
                for btch in tloader:
                    optimizer.zero_grad()
                    yhat = model(btch[0])
                    lss = loss(yhat, *btch[1:])

                    total = sum(lss.values())
                    total.backward()

                    optimizer.step()
                    tloss += total.item()

                # validation minibatches
                for btch in vloader:
                    with torch.no_grad():
                        yhat = model(btch[0])
                        lss = loss(yhat, *btch[1:])

                        total = sum(lss.values())
                        vloss += total.item()

                # update log
                pbar.set_postfix(
                    train=tloss / len(tloader),
                    validation=vloss / len(vloader) if len(vloader) else 0,
                )

                mlflow.log_metric("training-loss", tloss / len(tloader), step=e)
                mlflow.log_metric("validation-loss", vloss / len(vloader), step=e)

                qalpha = model.basis.qalpha(detach=True)
                for k in range(model.basis.K):
                    mlflow.log_metric(
                        f"basis-variance-{k}", 1 / qalpha.mean[k].item(), step=e
                    )
                    mlflow.log_metric(
                        f"basis-log-alpha-{k}",
                        model.basis.log_alpha[k].detach().item(),
                        step=e,
                    )
                    mlflow.log_metric(
                        f"basis-log-beta-{k}",
                        model.basis.log_beta[k].detach().item(),
                        step=e,
                    )

            # Save training results
            torch.save(model.state_dict(), output[0])
            torch.save(loss.state_dict(), output[1])

            # also save this specific version by id
            base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
            os.makedirs(base, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(base, "model.pt"))
            mlflow.log_artifact(os.path.join(base, "model.pt"), "model")
            torch.save(loss.state_dict(), os.path.join(base, "loss.pt"))
            mlflow.log_artifact(os.path.join(base, "loss.pt"), "loss")

    except Exception as e:
        base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
        os.makedirs(base, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(base, "model-error.pt"))
        torch.save(loss.state_dict(), os.path.join(base, "loss-error.pt"))
        raise e

rule lantern_cv:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}/model.pt",
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}/loss.pt"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8), meanEffectsInit=False),
            Phenotype.fromDataset(ds, dsget("K", 8))
        )

        # Setup training infrastructure
        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])
        tloader = DataLoader(train, batch_size=dsget("lantern/batch-size", default=8192))
        vloader = DataLoader(validation, batch_size=dsget("lantern/batch-size", default=8192))

        loss = model.loss(N=len(train), sigma_hoc=ds.errors is not None)

        if dsget("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        optimizer = Adam(loss.parameters(), lr=dsget("lantern/lr", default=0.01))

        mlflow.set_experiment("lantern cross-validation")

        # Run optimization
        try:
            with mlflow.start_run() as run:
                mlflow.log_param("dataset", wildcards.ds)
                mlflow.log_param("model", "lantern")
                mlflow.log_param("lr", dsget("lantern/lr", default=0.01))
                mlflow.log_param("cv", wildcards.cv)
                mlflow.log_param("batch-size", tloader.batch_size)

                pbar = tqdm(range(dsget("lantern/epochs", default=5000)),)
                best = np.inf
                for e in pbar:

                    # logging of loss values
                    tloss = 0
                    vloss = 0

                    # go through minibatches
                    for btch in tloader:
                        optimizer.zero_grad()
                        yhat = model(btch[0])
                        lss = loss(yhat, *btch[1:])

                        total = sum(lss.values())
                        total.backward()

                        optimizer.step()
                        tloss += total.item()

                    # validation minibatches
                    for btch in vloader:
                        with torch.no_grad():
                            yhat = model(btch[0])
                            lss = loss(yhat, *btch[1:])

                            total = sum(lss.values())
                            vloss += total.item()

                    # update log
                    pbar.set_postfix(
                        train=tloss / len(tloader),
                        validation=vloss / len(vloader) if len(vloader) else 0,
                    )

                    mlflow.log_metric("training-loss", tloss / len(tloader), step=e)
                    mlflow.log_metric("validation-loss", vloss / len(vloader), step=e)

                    qalpha = model.basis.qalpha(detach=True)
                    for k in range(model.basis.K):
                        mlflow.log_metric(f"basis-variance-{k}", 1 / qalpha.mean[k].item(), step=e)

                # Save training results
                torch.save(model.state_dict(), output[0])
                torch.save(loss.state_dict(), output[1])

                # also save this specific version by id
                base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
                os.makedirs(base, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(base, "model.pt"))
                mlflow.log_artifact(os.path.join(base, "model.pt"), "model")
                torch.save(loss.state_dict(), os.path.join(base, "loss.pt"))
                mlflow.log_artifact(os.path.join(base, "loss.pt"), "loss")

        except Exception as e:
            base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
            os.makedirs(base, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(base, "model-error.pt"))
            torch.save(loss.state_dict(), os.path.join(base, "loss-error.pt"))
            raise e

rule lantern_full:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/lantern/full{rerun,.*}/model.pt",
        "experiments/{ds}-{phenotype}/lantern/full{rerun,.*}/loss.pt"
    resources:
        gres="gpu:1",
        partition="batch",
    group: "train"
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8)),
            Phenotype.fromDataset(ds, dsget("K", 8)),
        )

        loss = model.loss(N=len(ds), sigma_hoc=ds.errors is not None)

        if dsget("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        # Setup training infrastructure
        tloader = DataLoader(ds, batch_size=dsget("lantern/batch-size", default=8192))

        optimizer = Adam(loss.parameters(), lr=dsget("lantern/lr", default=0.01))

        mlflow.set_experiment("lantern full")

        # Run optimization
        try:
            with mlflow.start_run() as run:
                mlflow.log_param("dataset", wildcards.ds)
                mlflow.log_param("model", "lantern")
                mlflow.log_param("lr", dsget("lantern/lr", default=0.01))
                mlflow.log_param("batch-size", tloader.batch_size)

                pbar = tqdm(range(dsget("lantern/epochs", default=5000)),)
                best = np.inf
                for e in pbar:

                    # logging of loss values
                    tloss = 0

                    # go through minibatches
                    for btch in tloader:
                        optimizer.zero_grad()
                        yhat = model(btch[0])
                        lss = loss(yhat, *btch[1:])

                        total = sum(lss.values())
                        total.backward()

                        optimizer.step()
                        tloss += total.item()

                    # update log
                    pbar.set_postfix(train=tloss / len(tloader),)

                    mlflow.log_metric("training-loss", tloss / len(tloader), step=e)

                    qalpha = model.basis.qalpha(detach=True)
                    for k in range(model.basis.K):
                        mlflow.log_metric(
                            f"basis-variance-{k}", 1 / qalpha.mean[k].item(), step=e
                        )

                # Save training results
                torch.save(model.state_dict(), output[0])
                torch.save(loss.state_dict(), output[1])

                # also save this specific version by id
                base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
                os.makedirs(base, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(base, "model.pt"))
                mlflow.log_artifact(os.path.join(base, "model.pt"), "model")
                torch.save(loss.state_dict(), os.path.join(base, "loss.pt"))
                mlflow.log_artifact(os.path.join(base, "loss.pt"), "loss")

        except Exception as e:
            base = os.path.join(os.path.dirname(output[0]), "runs", run.info.run_id)
            os.makedirs(base, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(base, "model-error.pt"))
            torch.save(loss.state_dict(), os.path.join(base, "loss-error.pt"))
            raise e

rule lantern_prediction:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}/model.pt"
    output:
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}/pred-val.csv"
    group: "predict"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        CUDA = dsget("lantern/prediction/cuda", default=True) and torch.cuda.is_available()

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8)),
            Phenotype.fromDataset(ds, dsget("K", 8)),
        )

        model.load_state_dict(torch.load(input[2], "cpu"))
        model.eval()

        if CUDA:
            model = model.cuda()

        def save(ofile, df, **kwargs):
            for k, t in kwargs.items():
                for i in range(t.shape[1]):
                    df["{}{}".format(k, i)] = t[:, i]

            df.to_csv(ofile, index=False)

        with torch.no_grad():
            save(
                output[0],
                df[df.cv == float(wildcards.cv)],
                **predictions(
                    ds.D,
                    model,
                    validation,
                    cuda=CUDA,
                    size=dsget("lantern/prediction/size", default=32),
                    pbar=dsget("lantern/prediction/pbar", default=True),
                ),
            )

rule lantern_cv_size:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}-n{n}/model.pt",
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}-n{n}/loss.pt"
    group: "train"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8), meanEffectsInit=False),
            Phenotype.fromDataset(ds, dsget("K", 8)),
        )

        # Setup training infrastructure
        train = Subset(
            ds,
            np.random.choice(
                np.where(df.cv != float(wildcards.cv))[0],
                int(wildcards.n),
                replace=False,
            ),
        )
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])
        tloader = DataLoader(
            train, batch_size=dsget("lantern/batch-size", default=8192)
        )
        vloader = DataLoader(
            validation, batch_size=dsget("lantern/batch-size", default=8192)
        )

        loss = model.loss(N=len(train), sigma_hoc=ds.errors is not None)

        if dsget("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        optimizer = Adam(loss.parameters(), lr=dsget("lantern/lr", default=0.01))

        mlflow.set_experiment(f"lantern cross-validation n={wildcards.n}")

        lr = dsget("lantern/lr", default=0.01)
        epochs = dsget("lantern/epochs", default=5000)

        train_lantern(
            input,
            output,
            wildcards,
            tloader,
            vloader,
            optimizer,
            loss,
            model,
            lr,
            epochs,
        )

rule lantern_prediction_size:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}-n{n}/model.pt"
    output:
        "experiments/{ds}-{phenotype}/lantern/cv{cv,\d+}-n{n}/pred-val.csv"
    group: "predict"
    resources:
        gres="gpu:1",
        partition="batch",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        CUDA = dsget("lantern/prediction/cuda", default=True) and torch.cuda.is_available()

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, dsget("K", 8)),
            Phenotype.fromDataset(ds, dsget("K", 8)),
        )

        model.load_state_dict(torch.load(input[2], "cpu"))
        model.eval()

        if CUDA:
            model = model.cuda()

        def save(ofile, df, **kwargs):
            for k, t in kwargs.items():
                for i in range(t.shape[1]):
                    df["{}{}".format(k, i)] = t[:, i]

            df.to_csv(ofile, index=False)

        with torch.no_grad():
            save(
                output[0],
                df[df.cv == float(wildcards.cv)],
                **predictions(
                    ds.D,
                    model,
                    validation,
                    cuda=CUDA,
                    size=dsget("lantern/prediction/size", default=32),
                    pbar=dsget("lantern/prediction/pbar", default=True),
                ),
            )

rule lantern_cv_k:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
    output:
        "experiments/{ds}-{phenotype}/lantern-K{k,\d+}/cv{cv,\d+}/model.pt",
        "experiments/{ds}-{phenotype}/lantern-K{k,\d+}/cv{cv,\d+}/loss.pt"
    resources:
        gres="gpu:1",
        partition="singlegpu",
        time = "24:00:00",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, int(wildcards.k), meanEffectsInit=False),
            Phenotype.fromDataset(ds, int(wildcards.k))
        )

        # Setup training infrastructure
        train = Subset(ds, np.where(df.cv != float(wildcards.cv))[0])
        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])
        tloader = DataLoader(
            train, batch_size=dsget("lantern/batch-size", default=8192)
        )
        vloader = DataLoader(
            validation, batch_size=dsget("lantern/batch-size", default=8192)
        )

        loss = model.loss(N=len(train), sigma_hoc=ds.errors is not None)

        if dsget("cuda", True):
            model = model.cuda()
            loss = loss.cuda()
            ds.to("cuda")

        optimizer = Adam(loss.parameters(), lr=dsget("lantern/lr", default=0.01))

        mlflow.set_experiment(f"lantern cross-validation K={wildcards.k}")

        lr = dsget("lantern/lr", default=0.01)
        epochs = dsget("lantern/epochs", default=5000)

        train_lantern(
            input,
            output,
            wildcards,
            tloader,
            vloader,
            optimizer,
            loss,
            model,
            lr,
            epochs,
        )


rule lantern_prediction_cv_k:
    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern-K{k,\d+}/cv{cv,\d+}/model.pt"
    output:
        "experiments/{ds}-{phenotype}/lantern-K{k,\d+}/cv{cv,\d+}/pred-val.csv"
    group: "predict"
    resources:
        gres="gpu:1",
        partition="singlegpu",
    run:
        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)

        CUDA = dsget("lantern/prediction/cuda", default=True) and torch.cuda.is_available()

        # Load the dataset
        df = pd.read_csv(input[0])
        ds = pickle.load(open(input[1], "rb"))

        validation = Subset(ds, np.where(df.cv == float(wildcards.cv))[0])

        # Build model and loss
        model = Model(
            VariationalBasis.fromDataset(ds, int(wildcards.k), meanEffectsInit=False),
            Phenotype.fromDataset(ds, int(wildcards.k))
        )

        model.load_state_dict(torch.load(input[2], "cpu"))
        model.eval()

        if CUDA:
            model = model.cuda()

        def save(ofile, df, **kwargs):
            for k, t in kwargs.items():
                for i in range(t.shape[1]):
                    df["{}{}".format(k, i)] = t[:, i]

            df.to_csv(ofile, index=False)

        with torch.no_grad():
            save(
                output[0],
                df[df.cv == float(wildcards.cv)],
                **predictions(
                    ds.D,
                    model,
                    validation,
                    cuda=CUDA,
                    size=dsget("lantern/prediction/size", default=32),
                    pbar=dsget("lantern/prediction/pbar", default=True),
                ),
            )
