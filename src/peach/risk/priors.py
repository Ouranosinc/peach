from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize, special, stats

DATADIR = Path(__file__).parent.parent / "data"

def load_sherwood_ecs():
    return pd.read_json(DATADIR / "sherwood_ecs.json").reindex()


def model_weights_from_sherwood(models=None, method="L2", lambda_=100):
    r"""Return model weights based on their climate sensitivity, weighted by the prior climate sensitivity
    distribution from Sherwood et al. (2020).

    The weights are found by a number of different methods, many based on iterative quantile matching estimators.

    One of these methods is the one used by Cannon 2024, where the metric to minimize is

    .. math::

       C = \frac{1}{N_T} \\sum_{\tau \\in T} (q_{\tau} - \\hat{q}_{\tau})^2 + \\lambda \frac{1}{N}\\sum_{i=1}^N \frac{w_i^2/w_0^2}{1 + w_i^2/w_0^2}

    where :math:`q_{\tau}` is the set of quantiles, :math:`q_{\tau}` is the weighted sample quantile, :math:`w_i` is the
    weight assigned to the ith model, :math:`w_0=1/N` and :math:`\\lambda` controls the penalty for non-zero weights.

    One issue with this approach is that for lambda=0, many of the weights are zero. There are cases where it might be
    preferable to keep as many models within the ensemble, in order to maximize the information content of the ensemble.
    Other methods are thus included that minimize the difference between the weighted and unweighted quantiles, while
    trying to keep the weights as uniform as possible.

    Parameters
    ----------
    models : list
        List of model names.
    method: {"L2", "L2Var", "Cannon", "L2SG", "L2D2", "kde"}
        Metric to minimize to estimate the weights.
        "Cannon" is the method described above.
        "L2" minimizes the L2 norm of the difference between the quantiles.
        "L2Var" adds to L2 the variance of the weights Var(w), normalized by 1+Var(w).
        "L2SGradient" adds to L2 the squared gradient of the weights , normalized by 1+(∇w)².
        "L2Laplacian" adds to L2 the Laplacian of the weights , normalized by 1+∇²w.
        "KLVar" minimizes the relative entropy between the theoretical distribution and the KDE estimated from the weighted sample, with a penalty based on the variance of the weights.
    lambda : float
        Importance given to the penalty term in the cost function. The higher, the smoother the weights.

    Returns
    -------
    weights : xr.DataArray
        Array of weights for each model.

    Notes
    -----
    L2 seems to struggle with low number of models (3)

    """  # noqa: RST306
    # Load prior distribution from Sherwood et al. (2020)
    prior_ = load_sherwood_ecs()
    pdf = xr.DataArray(prior_["pdf"], dims="ecs", coords={"ecs": prior_["ECS"]})
    cdf_ = (prior_["pdf"][99:].cumsum() * prior_["ECS"][99:].diff())[1:]
    ppf = xr.DataArray(prior_["ECS"][100:], dims="quantile", coords={"quantile": cdf_})

    # Load model ECS from Zelinka
    # Included ECS value for KIOST-ESM from http://dx.doi.org/10.1007/s12601-021-00001-7
    zelinka_ = (
        pd.read_json(DATADIR / "zelinka_ecs.json").reindex().sort_values("ECS")["ECS"]
    )

    # DataArray with ECS values
    ecs = xr.DataArray(zelinka_, dims="source_id")

    # Endpoints to reduce border effects
    endpoints = xr.DataArray([1.5, 7], dims="source_id", coords={"source_id": ["LOWER", "UPPER"]})

    # Initialize the weights
    w = xr.zeros_like(ecs)

    # Subset models
    subset = ecs.sel(source_id=list(set(models))) if models is not None else ecs
    model_sel = subset.source_id

    subset = xr.concat([subset, endpoints], dim="source_id")

    # Filter models
    t = xr.DataArray(np.arange(0.005, 1, 0.01), dims="quantile")

    q_ref = ppf.interp(quantile=t)

    def l2(w):
        """The mean difference squared between the empirical and theoretical quantiles."""
        w = xr.DataArray(w, dims="source_id", coords={"source_id": subset.source_id})
        q_est = subset.weighted(w).quantile(t)
        return ((q_est - q_ref) ** 2).mean()

    if method == "L2":
        cost = lambda w, ll: l2(w)

    elif method == "L2Var":
        cost = lambda w, ll: l2(w) + ll * np.var(w) / (1 + np.var(w))

    elif method == "Cannon":

        def cardinality(w):
            w02 = len(w) ** -2
            return ((w**2 / w02) / (1 + w**2 / w02)).mean()

        cost = lambda w, ll: l2(w) + ll * cardinality(w)

    elif method == "L2SGradient":

        def wcost(w):
            ww = xr.DataArray(w, dims="ecs", coords={"ecs": subset.data})
            c = (ww.differentiate("ecs") ** 2).mean()
            return c / (1 + c)

        cost = lambda w, ll: l2(w) + ll * wcost(w)

    elif method == "L2Laplacian":

        def wcost(w):
            ww = xr.DataArray(w, dims="ecs", coords={"ecs": subset.data})
            c = np.abs(ww.differentiate("ecs").differentiate("ecs")).mean()
            return c / (1 + c)

        cost = lambda w, ll: l2(w) + ll * wcost(w)

    elif method == "KLVar":
        q = pdf.isel(ecs=slice(100, None, 8))

        def cost(w, ll):
            kde = stats.gaussian_kde(subset.data, weights=w, bw_method=0.25)
            return special.rel_entr(kde(q.ecs), q).mean() + ll * np.var(w)

    else:
        raise ValueError("Unknown method")

    # Required to have non negative values
    bnds = tuple((0, 1) for x in subset)

    # Must sum to 1
    cons = {"type": "eq", "fun": lambda w: 1 - sum(w)}

    # w0 = np.ones(len(subset)) / len(subset)

    # Initial guess
    em = stats.gaussian_kde(subset)(subset)
    th = pdf.sel(ecs=subset, method="nearest")
    w0 = th / em
    w0 /= w0.sum()

    wi = optimize.minimize(
        cost, w0, method="SLSQP", bounds=bnds, constraints=cons, args=(lambda_,)
    )
    # print("L2: ", l2(wi.x))
    # print("Var: ", lambda_ * np.var(wi.x) / (1 + np.var(wi.x)))
    wi = xr.DataArray(wi.x, dims="source_id", coords={"source_id": subset.source_id})

    out = wi.sel(source_id=model_sel).reindex_like(w, fill_value=0)
    out.coords["ecs"] = ecs
    out.attrs["lambda"] = lambda_

    # Renormalize after removing LOWER and UPPER endpoints
    out /= out.sum()
    return out


def _merge_netcdf_likelihoods():
    import xarray as xr

    # Load likelihoods
    ll = {}
    for fn in (DATADIR / "ssp_likelihoods").glob("*.nc"):
        ds = xr.open_dataset(fn)
        ll[ds.attrs["source"]] = ds["likelihood"]

    ds = xr.concat(ll.values(), dim="source")
    ds = ds.assign_coords(source=list(ll.keys()))
    ds.attrs.pop("reference")
    ds.attrs.pop("source")
    return ds


def scenario_weights_from_iams(weights=None):
    """Use method from Huard et al. (2022) to compute weights for scenarios ssp126, ssp245, ssp370 and ssp585.

    Parameters
    ----------
    weights: dict
        Dictionary keyed by IAMs and values representing the weight given to each IAM. If None, give all IAMs equal
        weight. Keys correspond to results from the following papers:
        - CP16: Capellan-Perez et al. (2016)
        - CP20: Capellan-Perez et al. (2020)
        - FM15: Fyke and Matthews (2015)
        - LR21: Liu and Raftery (2021)
        - R17: Raftery et al. (2017)
        - S23: Sarofim et al. (2023)


    Notes
    -----
    Ma suggestion (DH) est d'assigner les poids suivants:
    w = {"FM15": 0.2, "CP16": 0.2, "CP20": 0.2, "LR21": 0.2, "S23": 0.2}
    parce que LR21 est une mise-à-jour de R17.
    """
    da = xr.open_dataarray(DATADIR / "ssp_likelihoods.nc", engine="h5netcdf")

    # Defaults for weights
    if weights is None:
        sources = list(da.source.data)
        weights = {s: 1 / len(sources) for s in sources}
    elif isinstance(weights, (list, tuple)):
        weights = {s: 1 / len(weights) for s in weights}

    # Convert dict to DataArray
    if isinstance(weights, dict):
        weights = xr.DataArray(
            list(weights.values()), coords={"source": list(weights.keys())}
        )

    # Compute weighted average of likelihoods for each SSP
    out = da.weighted(weights).mean("source")

    # Likelihoods are estimated every 10 years. Interpolate at yearly frequency to avoid jumps.
    t = xr.cftime_range("2015", "2101", freq="YE").to_datetimeindex(time_unit="us")
    return out.interp(time=t)


def number_of_members(da, missing=0.1, dim="variant_label") -> xr.DataArray:
    """Compute the number of members with valid data in a dataset."""
    # Count number of null entries along time.
    n = da.isnull().sum(dim="time")

    # Consider as valid only those with less than 10% of missing values.
    valid = n < missing * len(da.time)

    # Number of members
    return valid.sum(dim=dim)


def members(da):
    """Return weights related to the number of realizations (1/n) for a given model and scenario."""
    return 1 / number_of_members(da)


def weights(ds) -> xr.Dataset:
    """
    Return weights for the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions `source_id`, `experiment_id`, `variant_label` and `time` storing indicator
        values.

    Returns
    -------
    xr.Dataset
        Dataset with the same dimensions as `ds` storing the weights.
    """
    models = list(ds.source_id.values)
    exps = set(ds.experiment_id.values)
    if exps != {"ssp126", "ssp245", "ssp370", "ssp585"}:
        raise ValueError(
            "Experiments must be ssp126, ssp245, ssp370 and ssp585, otherwise, the weighting will be off."
        )

    mw = model_weights_from_sherwood(models, method="L2Var", lambda_=0.5)
    ew = scenario_weights_from_iams()

    # Align the time of the scenario weights on the dataset time index
    _, ew = xr.align(ds.time, ew, join="left")

    ew = ew.mean(dim="time")

    return ew * mw


def calcul_des_poids_pour_guillaume():
    path = DATADIR.parent.parent.parent / "data" / "Modèles retenus CMIP6 - INRS.xlsx"
    df = pd.read_excel(path).set_index("Model ID")
    df.drop(index="Total", inplace=True)
    df.drop(index="KIOST-ESM", inplace=True)

    out = []
    for col in df.columns:
        m = list(df[col].dropna().index)
        out.append(
            model_weights_from_sherwood(m, method="L2Var", lambda_=0.5).to_dataframe(
                name=col
            )[col]
        )

    return pd.concat(out, axis=1)


def graph_model_weights(w):
    """Create graphic showing the theoretical and empirical distribution of the PDF, CDF, and the weights."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(layout="constrained", figsize=(10, 6))
    fig.subplots_adjust(wspace=0.1, top=0.95, right=0.98)
    gs = GridSpec(2, 5, figure=fig)
    wax = fig.add_subplot(gs[:, 0])

    prior_ = pd.read_json(DATADIR / "sherwood_ecs.json").reindex()
    pdf = xr.DataArray(prior_["pdf"], dims="ecs", coords={"ecs": prior_["ECS"]})
    cdf = xr.DataArray(prior_["cdf"], dims="ecs", coords={"ecs": prior_["ECS"]})

    # Plot weights
    wax.barh(y=-np.arange(len(w)), width=w, height=0.5, color="red", clip_on=False)
    wax.set_yticks(-np.arange(len(w)))
    wax.set_yticklabels(w.source_id.values)
    wax.tick_params(axis="y", labelsize=6, length=0)
    wax.set_ylim(-len(w) + 1, 0)
    for spine in ["top", "right", "left"]:
        wax.spines[spine].set_visible(False)
    wax.spines["bottom"].set_position(("outward", 4))
    wax.axvline(1 / len(w), color="gray", lw=0.5)
    wax.set_xlabel("Weights")

    # CDF
    ax = fig.add_subplot(gs[0, 1:])
    ax.set_ylabel("CDF")
    ax.plot(cdf.ecs, cdf, color="k", label="Prior - Sherwood")
    #u = xr.DataArray(np.arange(0.025, 1, 0.025), dims="quantile")
    u = xr.DataArray(np.arange(0.001, 1, 0.001), dims="quantile")
    q_est = (
        xr.DataArray(w.ecs, dims="source_id", coords={"source_id": w.source_id})
        .weighted(w)
        .quantile(u)
    )
    ax.plot(q_est, u, color="red", label="Weighted CDF")

    # x = [-1, 0, 1, *q_est.data.tolist(), 7, 8, 9]
    # y = [0, 0, 0, *u.data.tolist(), 1, 1, 1]
    # we = [15, 10, 5, *np.ones(len(u)), 5, 10, 15]
    # spline = scipy.interpolate.UnivariateSpline(x, y, we, s=1E-2)

    # x = np.linspace(1.5, 5.5, 100)
    # ax.plot(x, spline(x), color="green", label="Spline")
    ax.legend()

    # PDF
    ax = fig.add_subplot(gs[1, 1:])
    # Plot prior
    ax.plot(pdf.ecs, pdf, color="k", label="Prior - Sherwood")
    ax.set_xlabel("ECS (°K)")
    ax.set_ylabel("PDF")

    # Plot weighted pdf from diff of CDF
    # ax.plot(x,spline.derivative()(x), color="green", label="Weighted CDF")

    # Plot weighted pdf with KDE
    kde = stats.gaussian_kde(w.ecs, weights=w)
    ax.plot(pdf.ecs, kde(pdf.ecs), color="orange", label="Weighted KDE PDF")

    # Plot weights
    ax.bar(w.ecs, w, width=0.1, color="red", alpha=0.5, label="Model weights")
    ax.legend()
    return fig


def make_figs(outdir=".", lambda_=0):
    outdir = Path(outdir)
    import matplotlib.pyplot as plt

    for method in ["L2", "L2Var", "Cannon", "L2SGradient", "L2Laplacian", "KLVar"]:
        w = model_weights_from_sherwood(method=method, lambda_=lambda_)
        fig = graph_model_weights(w)
        fig.savefig(outdir / f"weights_{method}_l{lambda_:.2f}.png")
        plt.close()
