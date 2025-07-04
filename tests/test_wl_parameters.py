import os

import numpy as np
import pytest
import xarray as xr
import xclim as xc
from scipy.stats import norm

from portail_ing.frontend.parameters import Analysis, HazardMatrix
from portail_ing.frontend.wl_parameters import (
    IndicatorObsWL,
    IndicatorRefWL,
    IndicatorSimWL,
    scale_pareto,
)
from portail_ing.risk import bootstrap


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping this test on GitHub CI"
)
def test_IndicatorsWL(wl_pot_obs, wl_pot_sim):

    dao = xr.open_dataarray(wl_pot_obs, engine="zarr")
    o = IndicatorObsWL(data=dao, period=(1970, 2010))

    das = xr.open_dataarray(wl_pot_sim, engine="zarr")
    ref = IndicatorRefWL(data=das, obs=o)

    fut = IndicatorSimWL(
        data=das, obs=o, period=(2040, 2070), model_weights=ref.param.model_weights
    )

    assert ref.isf(0.1) > 0


class TestIndicatorObsWL:
    def test_basic(self, synthetic_ewl_ds):
        wl, wl_pot, sl, stn_thresh = synthetic_ewl_ds

        ind = IndicatorObsWL(data=wl_pot, period=(1970, 2010))
        # assert ind.title == "wl_pot"
        assert ind.isf(0.9).name == "isf"
        np.testing.assert_allclose(ind.sf(ind.isf(0.1)), 0.1)

        # The likelihood of upper tail values should increase as the number of peaks per year increases
        x = ind.isf(0.1)
        y1 = ind.pdf(x)
        ind.peaks_per_yr = 10
        y2 = ind.pdf(x)
        assert y2 > y1

        # Check that changing the period  updates the params
        obs = IndicatorObsWL(data=wl_pot, period=(1960, 1990))
        dparams1 = obs.dparams
        obs.period = (1995, 2014)
        dparams2 = obs.dparams
        assert not np.array_equal(dparams1.values, dparams2.values)


class TestIndicatorSimWL:
    def test_basic(self, synthetic_ewl_ds):
        wl, wl_pot, sl, stn_thresh = synthetic_ewl_ds
        wl_pot.attrs["peaks_per_yr"] = 1.5
        obs = IndicatorObsWL(data=wl_pot, period=(1970, 2010))
        ind = IndicatorSimWL(obs=obs, data=sl, period=(2020, 2050))
        # assert ind.title == "wl_pot"
        assert ind.isf(0.1).name == "isf"
        np.testing.assert_allclose(ind.sf(ind.isf(0.1)), 0.1, atol=1e-2)

        # Check that a change in period updates params.
        dparams1 = ind.dparams
        ind.period = (2070, 2100)
        dparams2 = ind.dparams
        assert not np.array_equal(dparams1.values, dparams2.values)

        # Smoke test for the pdf
        x = ind.isf(0.1)
        y1 = ind.pdf(x)
        assert y1 > 0


class TestAnalysisWL:
    def test_basic(self, synthetic_ewl_ds, station_data):
        wl, wl_pot, sl, stn_thresh = synthetic_ewl_ds
        a = Analysis(
            ds={"obs": {"123": wl_pot}, "sim": {"123": sl}},
        )
        hm = HazardMatrix(analysis=a)
        assert len(hm.matrix) == 1


def test_scale_pareto():
    d = norm(0, 1)

    assert scale_pareto(d.sf, "sf", 0, 2) == 1
    assert scale_pareto(d.isf, "isf", 1, 2) == 0

    assert np.isclose(
        scale_pareto(d.sf, "sf", scale_pareto(d.isf, "isf", 0.1, 2), 2), 0.1, atol=1e-6
    )


def test_dparams_with_slr(synthetic_ewl_ds):
    wl, wl_pot, sl, stn_thresh = synthetic_ewl_ds
    ref_period = (1995, 2014)
    fut_period = (2070, 2100)

    obs = IndicatorObsWL(data=wl_pot, period=ref_period)
    ind = IndicatorSimWL(obs=obs, data=sl, period=fut_period)
    ind.sample_size = ind.obs.data.size
    dparams = ind.fit("genpareto", period=fut_period)
    loc = (
        dparams.sel(experiment_id="ssp585", dparams="loc")
        .quantile(dim="sample", q=0.5)
        .item()
    )
    loc_diff = loc - obs.dparams.sel(dparams="loc").item()

    # Calculate expected difference in loc from slr data.
    expected_diff = (
        sl.sel(experiment_id="ssp585")
        .interp(time=str(int(np.mean(fut_period))))
        .sel(quantile=0.5, method="nearest")
        .item()
    )
    assert np.isclose(
        expected_diff, loc_diff, atol=0.01
    )  # Tolerance for random noise in slr data and sampling with replacement.


def test_wl_mixture_replace_weights(synthetic_ewl_ds):
    wl, wl_pot, sl, stn_thresh = synthetic_ewl_ds
    thresh = 4

    obs = IndicatorObsWL(data=wl_pot, period=(1995, 2014))
    ind = IndicatorSimWL(obs=obs, data=sl, period=(2070, 2100))
    ind.weights.loc[
        ind.weights["experiment_id"].isin(["ssp126", "ssp370", "ssp585"])
    ] = 0
    ind.weights.loc[ind.weights["experiment_id"] == "ssp245"] = (
        1 / ind.weights.sizes["sample"]
    )
    aep = ind._dist_method("sf", thresh).item()

    sf_check = lambda x: xc.indices.stats.dist_method(
        function="sf", fit_params=ind.dparams, arg=x
    )
    aep_check = scale_pareto(sf_check, "sf", thresh, ind.obs.peaks_per_yr).quantile(
        dim="sample", q=0.5
    )

    assert np.all(
        np.isclose(np.array([0.0, 1.0, 0.0, 0.0]), ind.weights.sum(dim="sample").data)
    )
    assert np.isclose(ind.weights.sum(), 1)
    assert np.isclose(aep, aep_check.sel(experiment_id="ssp245").item(), atol=0.06)


def test_wl_mixture_bounds(synthetic_ewl_ds):
    wl, wl_pot, sl, stn_thresh = synthetic_ewl_ds
    obs = IndicatorObsWL(data=wl_pot, period=(1995, 2014))
    ind = IndicatorSimWL(obs=obs, data=sl, period=(2070, 2100))

    sf = lambda x: xc.indices.stats.dist_method(
        function="sf", fit_params=ind.dparams, arg=x
    )

    for thresh in [2.5, 3, 4]:
        mix = ind._dist_method("sf", thresh).item()
        ssps = (
            scale_pareto(sf, "sf", thresh, ind.obs.peaks_per_yr)
            .mean(dim="sample")
            .values
        )

        assert np.greater(mix + 1e-2, ssps.min())
        assert np.less(mix - 1e-2, ssps.max())


def test_wl_bootstrap(synthetic_ewl_ds):

    wl, wl_pot, sl, stn_thresh = synthetic_ewl_ds
    fut_period = (2070, 2100)
    obs = IndicatorObsWL(data=wl_pot, period=(1995, 2014))
    ind = IndicatorSimWL(obs=obs, data=sl, period=fut_period)

    q_fromdata = (
        sl.sel(experiment_id="ssp585")
        .sel(quantile=0.5)
        .interp(time=str(int(np.mean(fut_period))))
        .item()
    )
    q_frombootstrap = (
        bootstrap.from_quantile(ind._sample(period=fut_period), 2000)
        .sel(experiment_id="ssp585")
        .quantile(dim="sample", q=0.5)
        .item()
    )

    assert np.isclose(q_fromdata, q_frombootstrap, atol=5e-4)


def test_wl_experiment_percentiles(synthetic_ewl_ds):

    wl, wl_pot, sl, stn_thresh = synthetic_ewl_ds
    fut_period = (2070, 2100)
    obs = IndicatorObsWL(data=wl_pot, period=(1995, 2014))
    ind = IndicatorSimWL(obs=obs, data=sl, period=fut_period)

    q_fromdata = (
        sl.sel(experiment_id="ssp585")
        .sel(quantile=0.5)
        .interp(time=str(int(np.mean(fut_period))))
        .item()
    )
    q_experiment_percentiles = (
        ind.experiment_percentiles([50])[f"{ind.data.name}_p50"]
        .interp(time=str(int(np.mean(fut_period))))
        .sel(experiment_id="ssp585")
        - ind.obs.stn_thresh
    )

    assert np.isclose(q_fromdata, q_experiment_percentiles, atol=1e-5)
