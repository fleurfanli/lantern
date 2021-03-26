from torch.utils.data import DataLoader
import numpy as np
import gpytorch
import torch
from tqdm import tqdm

# from src.analyze.grad import gradient, laplacian
# from src.model import LatentLinearGPBayes


def predictions(
    D, model, dataset, size=32, cuda=False, pbar=False, uncertainty=False, dimScan=False
):

    embed = hasattr(model, "L")
    diffops = False  # isinstance(model, LatentLinearGPBayes) and model.D == 1

    # prep for predictions
    loader = DataLoader(dataset, size)
    yhat = torch.zeros(len(dataset), D)
    y = torch.zeros(len(dataset), D)
    noise = torch.zeros(len(dataset), D)
    lp = torch.zeros(len(dataset), D)
    yhat_std = torch.zeros(len(dataset), D)
    if dimScan:
        yhat_scan = torch.zeros(len(dataset), model.L * D)

    if diffops:
        grad_mu = torch.zeros(len(dataset), model.L)
        grad_var = torch.zeros(len(dataset), model.L)
        lapl_mu = torch.zeros(len(dataset), 1)
        lapl_var = torch.zeros(len(dataset), 1)
        z0 = torch.from_numpy(model.landscape._get_induc())

        # magic number, consider changing
        dims = sum(
            (model.embed.log_beta.exp() / (model.embed.log_alpha.exp() - 1)) > 1e-2
        )
        dims = model.embed.variance_order[:dims]

    if embed:
        z = torch.zeros(len(dataset), model.L)

    if cuda:
        yhat = yhat.cuda()
        y = y.cuda()
        lp = lp.cuda()
        if embed:
            z = z.cuda()
        yhat_std = yhat_std.cuda()

        if diffops:
            grad_mu = grad_mu.cuda()
            grad_var = grad_var.cuda()
            lapl_mu = lapl_mu.cuda()
            lapl_var = lapl_var.cuda()

            z0 = z0.cuda()

    # loop over data and generate predictions
    i = 0
    loop = tqdm(loader) if pbar else loader
    for btch in loop:
        _x, _y = btch[:2]
        _x = _x.float()
        if cuda:
            _x = _x.cuda()
            _y = _y.cuda()

        with torch.no_grad():
            _yh = model(_x)

            if embed:
                _z = model.embed(_x)
                if isinstance(_z, tuple):
                    _z = _z[0]

            if diffops:
                _grad = gradient(model.landscape, _z, z0)
                _lapl = laplacian(model.landscape, _z, z0, dims=dims)

            if uncertainty:
                Nsamp = 50
                tmp = torch.zeros(Nsamp, *_y.shape)
                model.train()
                for n in range(Nsamp):
                    f, _ = model(_x)
                    samp = f.sample()
                    if samp.ndim == 1:
                        samp = samp[:, None]

                    tmp[n, :, :] = samp

                yhat_std[i : len(_y) + i, :] = tmp.std(axis=0)

                model.eval()

            # check prediction accuracy as a function of available
            # latent dimensions
            if dimScan:
                _z = model.basis(_x)
                if isinstance(_z, tuple):
                    _z = _z[0]

                for l in range(model.L):
                    _zz = torch.zeros_like(_z)

                    # copy lth first dimensions
                    _zz[:, model.embed.variance_order[:l]] = _z[
                        :, model.embed.variance_order[:l]
                    ]
                    _yyh = model.landscape(_zz)

                    # get to actual prediction
                    if isinstance(_yyh, tuple):
                        _yyh = _yyh[0]

                    if isinstance(_yyh, gpytorch.distributions.MultivariateNormal):
                        _yyh = _yyh.mean

                    if _yyh.ndim == 1:
                        _yyh = _yyh[:, None]

                    yhat_scan[i : len(_y) + i, l * D : (l + 1) * D] = _yyh

        # filter out extra output
        if isinstance(_yh, tuple):
            _yh = _yh[0]

        # need to get a mean prediction, and we can get logprob
        if isinstance(_yh, gpytorch.distributions.MultivariateNormal):
            # # get predictive observation likelihood
            # _yh = model.mll.likelihood(_yh)

            # convert to 1d for individual observations
            # norm = torch.distributions.Normal(
            #     _yh.mean.view(-1, D), torch.sqrt(_yh.variance.view(-1, D))
            # )

            # # update values
            # _lp = norm.log_prob(_y).detach()
            # _lp = _lp.view(-1, D)
            # lp[i : len(_y) + i, :] = _lp
            _yh = _yh.mean

        _y = _y.view(-1, D)
        _yh = _yh.view(-1, D)
        y[i : len(_y) + i, :] = _y
        yhat[i : len(_y) + i, :] = _yh

        if embed:
            z[i : len(_y) + i, :] = _z

        if diffops:
            grad_mu[i : len(_y) + i, :] = _grad[0][:, :, 0]
            grad_var[i : len(_y) + i, :] = _grad[1][
                :, np.arange(model.L), np.arange(model.L)
            ]
            lapl_mu[i : len(_y) + i, 0] = _lapl[0]
            lapl_var[i : len(_y) + i, 0] = _lapl[1]

        # grab noise if available
        if len(btch) > 2:
            _n = btch[2]
            if cuda:
                _n = _n.cuda()
            noise[i : len(_n) + i, :] = _n

        i += len(_x)

    # prep for returning
    y = y.cpu().numpy()
    yhat = yhat.cpu().numpy()
    lp = lp.cpu().numpy()
    noise = noise.cpu().numpy()

    if embed:
        if hasattr(model.embed, "variance_order"):
            z = z[:, model.embed.variance_order]
        z = z.cpu().numpy()

    if diffops:
        grad_mu = grad_mu.cpu().numpy()
        grad_var = grad_var.cpu().numpy()
        lapl_mu = lapl_mu.cpu().numpy()
        lapl_var = lapl_var.cpu().numpy()

    # ret = dict(y=y, yhat=yhat, logprob=lp, noise=noise,)
    ret = dict(y=y, yhat=yhat, noise=noise,)

    if embed:
        ret["z"] = z

    if diffops:
        ret["grad_mu"] = grad_mu
        ret["grad_var"] = grad_var
        ret["lapl_mu"] = lapl_mu
        ret["lapl_var"] = lapl_var

    if uncertainty:
        ret["yhat_std"] = yhat_std.cpu().numpy()

    if dimScan:
        for l in range(model.L):
            ret[f"yhat_d{l}_"] = yhat_scan[:, l * D : (l + 1) * D].cpu().numpy()

    return ret