import numpy as np
import xarray as xr
import xclim as xc

from peach.risk import xmixture


def test_weighted_indicators(synthetic_dataset_fut):
    from peach.risk.priors import weights

    ds = synthetic_dataset_fut
    dims = ("source_id", "experiment_id")
    da = ds["00"].unstack("realization")
    w = weights(da)

    dparams = xc.indices.stats.fit(
        da.stack(s=["time", "variant_label"]), dist="uniform", dim="s"
    )

    mix = xmixture.XMixtureDistribution(dparams.stack(r=dims), weights=w.stack(r=dims))
    out = mix.pdf([0.3, 0.4, 0.5])
    np.testing.assert_array_almost_equal(out, 1, decimal=1)


def test_xmixture_distribution():
    weights = xr.DataArray(
        data=[[0.15, 0.35], [0.1, 0.4]],
        dims=(
            "a",
            "b",
        ),
        coords={"a": [0, 1], "b": ["A", "B"]},
    )

    mu = xr.DataArray(
        data=[[0, 3], [2, 5]], dims=("a", "b"), coords={"a": [0, 1], "b": ["A", "B"]}
    )

    data = (
        xr.DataArray(np.random.normal(size=(2, 2, 200)), dims=("a", "b", "time")) + mu
    )

    mix = xmixture.XMixtureDistribution.from_data(
        data, weights=weights, distribution="norm"
    )
    exp = (mu * weights).sum()
    np.testing.assert_almost_equal(mix.ppf(0.5), exp, decimal=1)

    # non normalized weights
    nn_mix = xmixture.XMixtureDistribution.from_data(
        data, weights=weights * 2, distribution="norm"
    )

    np.testing.assert_almost_equal(nn_mix.ppf(0.5), exp, decimal=1)
