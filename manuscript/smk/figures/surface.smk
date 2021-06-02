rule surface:
    """
    Surface plot of lantern model.
    """

    input:
        "data/processed/{ds}.csv",
        "data/processed/{ds}-{phenotype}.pkl",
        "experiments/{ds}-{phenotype}/lantern/full/model.pt"
    output:
        "figures/{ds}-{phenotype}/{target}/surface.png"

    run:

        def dsget(pth, default):
            """Get the configuration for the specific dataset"""
            return get(config, f"{wildcards.ds}/{pth}", default=default)
        
        alpha = get(config, f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/alpha", default=0.01)
        raw = get(config, f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/raw", default=None)
        log = get(config, f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/log", default=False)
        p = get(config, f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/p", default=0)
        image = get(config, f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/image", default=False)
        scatter = get(config, f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/scatter", default=True)
        mask = get(config, f"figures/surface/{wildcards.ds}-{wildcards.phenotype}/{wildcards.target}/mask", default=False)

        df, ds, model = util.load_run(wildcards.ds, wildcards.phenotype, "lantern", "full", dsget("K", 8))
        model.eval()

        z, fmu, fvar, Z1, Z2, y, Z = util.buildLandscape(
            model,
            ds,
            mu=df[raw].mean() if raw is not None else 0,
            std=df[raw].std() if raw is not None else 1,
            log=log,
            p=p,
            alpha=alpha,
        )

        fig, norm, cmap, vrange = util.plotLandscape(
            z,
            fmu,
            fvar,
            Z1,
            Z2,
            log=log,
            image=image,
            mask=mask,
        )

        if scatter:
            
            plt.scatter(
                z[:, 0],
                z[:, 1],
                c=y,
                alpha=0.4,
                rasterized=True,
                vmin=vrange[0],
                vmax=vrange[1],
                norm=mpl.colors.LogNorm(vmin=vrange[0], vmax=vrange[1],) if log else None,
                s=0.3,
            )

            # reset limits
            plt.xlim(
                np.quantile(z[:, 0], alpha / 2),
                np.quantile(z[:, 0], 1 - alpha / 2),
            )
            plt.ylim(
                np.quantile(z[:, 1], alpha / 2),
                np.quantile(z[:, 1], 1 - alpha / 2),
            )


        plt.savefig(output[0], bbox_inches="tight", verbose=False)