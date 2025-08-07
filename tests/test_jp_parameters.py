import numpy as np
import pytest
import xclim as xc

pytest.importorskip("copulae")

from peach.frontend.jp_parameters import (  # noqa: E402
    IndicatorObsJP,
    IndicatorObsPRCOND,
    IndicatorObsPRPOT,
    IndicatorObsWLCOND,
    IndicatorSimJP,
    IndicatorSimPRCOND,
    IndicatorSimPRPOT,
    IndicatorSimWLCOND,
)
from peach.frontend.wl_parameters import (  # noqa: E402
    IndicatorObsWL,
    IndicatorSimWL,
)
from peach.risk.xmixture import XMixtureDistribution  # noqa: E402


@pytest.mark.parametrize("synthetic_jp_ds", ["pr_wlcond"], indirect=True)
def test_IndicatorObsWLCOND(synthetic_jp_ds):
    _, wl_cond_backend = synthetic_jp_ds
    marg = IndicatorObsWLCOND(data=wl_cond_backend, period=(1960, 2010))

    ### - TEST BASIC
    assert (
        marg.long_name == "Water level extremes conditional on precipitation extremes"
    )
    assert marg.dparams.attrs["ex_type"] == "Conditional Extremes"
    assert np.isclose(marg.sf(marg.isf(0.9)), 0.9)
    assert np.isclose(marg.cdf(marg.ppf(0.9)), 0.9)
    assert marg.isf(0.5).name == "isf"

    # Check that changing the period updates the params
    obs = IndicatorObsWLCOND(data=wl_cond_backend, period=(1960, 2010))
    dparams1 = obs.dparams
    obs.period = (1995, 2014)
    dparams2 = obs.dparams
    assert not np.array_equal(dparams1.values, dparams2.values)

    ### - TESTS DISTRIBUTION SELECTION (IndicatorObsDA)
    # Check expected distribution was selected
    assert marg.dist == "norm"
    # Check bic is lower for the expected distribution
    assert marg._bic(period=(1960, 2010), dist="norm") < marg._bic(
        period=(1960, 2010), dist="gamma"
    )
    # Check distribution can be assigned
    marg = IndicatorObsWLCOND(data=wl_cond_backend, period=(1960, 2010), dist="gamma")
    assert marg.dist == "gamma"
    # Check that a change in metric triggers a change in metrics
    marg.metric = "aic"
    assert marg.metrics_da.attrs.get("long_name") == "Akaike Information Criterion"

    ### - TEST FOR CONDITIONAL EXTREMES (i.e., no scale_pareto)
    # Check that likelihoods are not affected by peaks_per_yr
    assert not hasattr(marg, "peaks_per_yr")
    marg.peaks_per_yr = 4
    y1 = marg.sf(10)
    z1 = marg.pdf(10)
    marg.peaks_per_yr = 8
    y2 = marg.sf(10)
    z2 = marg.pdf(10)
    assert y2 == y1
    assert z2 == z1


@pytest.mark.parametrize("synthetic_jp_ds", ["pr_wlcond"], indirect=True)
def test_IndicatorObsPRPOT(synthetic_jp_ds):
    pr_pot_backend, _ = synthetic_jp_ds
    marg = IndicatorObsPRPOT(data=pr_pot_backend, period=(1970, 2010))

    ### - TEST BASIC
    assert marg.long_name == "Precipitation peaks over threshold"
    assert marg.dparams.attrs["ex_type"] == "Peaks over threshold"
    assert np.isclose(marg.sf(marg.isf(0.9)), 0.9)
    assert marg.isf(0.5).name == "isf"

    # Check that changing the period does not update the params
    obs = IndicatorObsPRPOT(data=pr_pot_backend, period=(1960, 1990))
    dparams1 = obs.dparams
    obs.period = (1995, 2014)
    dparams2 = obs.dparams
    assert np.array_equal(dparams1.values, dparams2.values)

    ### - TESTS FOR POT EXTREMES (i.e., scale_pareto)
    assert marg.dist == "genpareto"
    assert np.all(marg.data > marg.stn_thresh)

    # The likelihood of upper tail values should increase as the number of peaks per year increases
    marg.peaks_per_yr = 4
    y1 = marg.sf(10)
    z1 = marg.pdf(10)
    marg.peaks_per_yr = 8
    y2 = marg.sf(10)
    z2 = marg.pdf(10)
    assert y2 == y1 * 2
    assert z2 > z1

    # Ensure cannot be run with other distributions
    pytest.raises(
        ValueError,
        lambda: IndicatorObsPRPOT(
            data=pr_pot_backend, period=(1970, 2010), dist="uniform"
        ),
    )


@pytest.mark.parametrize("synthetic_jp_ds", ["wl_prcond"], indirect=True)
def test_IndicatorObsPRCOND(synthetic_jp_ds):
    _, pr_cond_backend = synthetic_jp_ds
    marg = IndicatorObsPRCOND(data=pr_cond_backend, period=(1970, 2010))

    assert (
        marg.long_name == "Precipitation extremes conditional on water level extremes"
    )
    assert marg.dparams.attrs["ex_type"] == "Conditional Extremes"
    assert np.isclose(marg.sf(marg.isf(0.9)), 0.9)
    assert np.isclose(marg.cdf(marg.ppf(0.9)), 0.9)
    assert marg.isf(0.5).name == "isf"

    # Check that changing the period does not update the params
    obs = IndicatorObsPRCOND(data=pr_cond_backend, period=(1960, 1990))
    dparams1 = obs.dparams
    obs.period = (1995, 2014)
    dparams2 = obs.dparams
    assert np.array_equal(dparams1.values, dparams2.values)

    ### - TESTS DISTRIBUTION SELECTION (IndicatorObsDA)
    # Check expected distribution was selected
    assert marg.dist == "norm"
    # Check bic is lower for the expected distribution
    assert marg._bic(period=(1960, 2010), dist="norm") < marg._bic(
        period=(1960, 2010), dist="gamma"
    )
    # Check distribution can be assigned
    marg = IndicatorObsPRCOND(data=pr_cond_backend, period=(1960, 2010), dist="gamma")
    assert marg.dist == "gamma"
    # Check that a change in metric triggers a change in metrics
    marg.metric = "aic"
    assert marg.metrics_da.attrs.get("long_name") == "Akaike Information Criterion"

    ### - TEST FOR CONDITIONAL EXTREMES (i.e., no scale_pareto)
    # Check that likelihoods are not affected by peaks_per_yr
    assert not hasattr(marg, "peaks_per_yr")
    marg.peaks_per_yr = 4
    y1 = marg.sf(10)
    z1 = marg.pdf(10)
    marg.peaks_per_yr = 8
    y2 = marg.sf(10)
    z2 = marg.pdf(10)
    assert y2 == y1
    assert z2 == z1


@pytest.mark.parametrize("synthetic_jp_ds", ["pr_wlcond"], indirect=True)
def test_IndicatorSimWLCOND(synthetic_jp_ds, synthetic_ewl_ds):
    _, _, sl, _ = synthetic_ewl_ds
    pr_pot_backend, wl_cond_backend = synthetic_jp_ds
    marg = IndicatorObsWLCOND(data=wl_cond_backend, period=(1960, 2010))
    sim = IndicatorSimWLCOND(obs=marg, data=sl, period=(2070, 2100))

    ### - TEST BASIC
    assert sim.long_name == "Water level extremes conditional on precipitation extremes"
    assert sim.dparams.attrs["ex_type"] == "Conditional Extremes"
    assert np.isclose(sim.sf(sim.isf(0.1)), 0.1, atol=1e-2)
    assert np.isclose(sim.cdf(sim.ppf(0.9)), 0.9, atol=1e-2)
    assert sim.isf(0.1) > marg.isf(0.1)
    # Check expected distribution was passed on
    assert sim.dist == "norm"
    assert sim.isf(0.5).name == "isf"

    # Check that a change in period updates params.
    sim = IndicatorSimWLCOND(obs=marg, data=sl, period=(2070, 2100))
    dparams1 = sim.dparams
    sim.period = (2020, 2050)
    dparams2 = sim.dparams
    assert not np.array_equal(dparams1.values, dparams2.values)

    ### - TEST FOR CONDITIONAL EXTREMES (i.e., no scale_pareto)
    # Check that likelihoods are not affected by peaks_per_yr
    assert not hasattr(sim, "peaks_per_yr")
    sim.peaks_per_yr = 4
    y1 = sim.sf(10)
    z1 = sim.pdf(10)
    sim.peaks_per_yr = 8
    y2 = sim.sf(10)
    z2 = sim.pdf(10)
    assert y2 == y1
    assert z2 == z1

    #### - TEST FOR SEA-LEVEL RISE & LOC
    sim = IndicatorSimWLCOND(obs=marg, data=sl, period=(2070, 2100))
    sim.sample_size = sim.obs.data.size
    loc = (
        sim.dparams.sel(experiment_id="ssp585", dparams="loc")
        .quantile(dim="sample", q=0.5)
        .item()
    )
    loc_diff = loc - sim.obs.dparams.sel(dparams="loc").item()

    # Calculate expected difference in loc from slr data.
    expected_diff = (
        sl.sel(experiment_id="ssp585")
        .interp(time=str(int(np.mean((2070, 2100)))))
        .sel(quantile=0.5, method="nearest")
        .item()
    )
    assert np.isclose(
        expected_diff, loc_diff, atol=1e-1
    )  # Tolerance for random noise in slr data and sampling with replacement.

    #### - test_wl_mixture_replace_weights
    sim.weights.loc[
        sim.weights["experiment_id"].isin(["ssp126", "ssp370", "ssp585"])
    ] = 0
    sim.weights.loc[sim.weights["experiment_id"] == "ssp245"] = (
        1 / sim.weights.sizes["sample"]
    )
    assert sim.weights.sum().item() == 1
    # thresh = 3
    # aep = sim.sf(thresh).item()

    # aep_check = xc.indices.stats.dist_method(
    #     function="sf", fit_params=sim.dparams.sel(experiment_id="ssp245"), arg=aep
    # ).quantile(dim="sample", q=0.5)

    assert np.all(
        np.isclose(np.array([0.0, 1.0, 0.0, 0.0]), sim.weights.sum(dim="sample").data)
    )
    assert np.isclose(sim.weights.sum(), 1)


@pytest.mark.parametrize("synthetic_jp_ds", ["pr_wlcond"], indirect=True)
def test_IndicatorSimPRPOT(synthetic_jp_ds, synthetic_ds_daily):
    pr_sim = synthetic_ds_daily
    fut_period = (2070, 2100)
    pr_pot_backend, wl_cond_backend = synthetic_jp_ds
    obs = IndicatorObsPRPOT(data=pr_pot_backend, period=(1995, 2014))
    sim = IndicatorSimPRPOT(obs=obs, data=pr_sim, period=fut_period)

    ### - TEST BASIC
    assert sim.long_name == "Precipitation peaks over threshold"
    assert np.isclose(sim.sf(sim.isf(0.1)), 0.1, atol=1e-1)
    assert sim.isf(0.1) > obs.isf(0.1)
    assert sim.isf(0.5).name == "isf"

    ### - TESTS FOR POT EXTREMES (i.e., scale_pareto)
    assert sim.dparams.attrs["ex_type"] == "Peaks over threshold"
    assert sim.dist == "genpareto"
    assert np.all(sim.obs.data.min() > sim.obs.stn_thresh)
    # The likelihood of upper tail values should increase as the number of peaks per year increases
    sim.obs.peaks_per_yr = 4
    y1 = sim.sf(10)
    z1 = sim.pdf(10)
    sim.obs.peaks_per_yr = 8
    y2 = sim.sf(10)
    z2 = sim.pdf(10)
    assert y2 == y1 * 2
    assert z2 > z1

    # Check that a change in period updates params.
    sim = IndicatorSimPRPOT(obs=obs, data=pr_sim, period=fut_period)
    dparams1 = sim.dparams
    sim.period = (2020, 2050)
    dparams2 = sim.dparams
    assert not np.array_equal(dparams1.values, dparams2.values)

    # Ensure cannot be run with other distributions
    pytest.raises(
        ValueError,
        lambda: IndicatorSimPRPOT(
            obs=obs, data=pr_sim, period=(2050, 2080), dist="uniform"
        ),
    )

    #### - TEST FOR PRECIPITATION DELTA
    sim = IndicatorSimPRPOT(obs=obs, data=pr_sim, period=fut_period)
    loc1 = sim.obs.dparams.sel(dparams="loc").item()
    loc2 = (
        sim.dparams.sel(
            dparams="loc", experiment_id="ssp585", source_id="HadGEM3-GC31-MM"
        )
        .median(dim="sample")
        .item()
    )
    loc_delta = loc2 / loc1

    ref = (
        pr_sim.sel(
            experiment_id="ssp585",
            source_id="HadGEM3-GC31-MM",
            variant_label="r1i1p1f1",
        )
        .sel(time=slice("1995-01-01", "2014-12-31"))
        .quantile(0.95, dim="time")
        .item()
    )
    fut = (
        pr_sim.sel(
            experiment_id="ssp585",
            source_id="HadGEM3-GC31-MM",
            variant_label="r1i1p1f1",
        )
        .sel(time=slice("2070-01-01", "2100-12-31"))
        .quantile(0.95, dim="time")
        .item()
    )
    expected_delta = fut / ref
    assert np.isclose(expected_delta, loc_delta, atol=1e-1)

    #### - TEST FOR WEIGHTS
    assert np.isclose(sim.weights.sum().item(), 1)
    assert set(sim.weights.dims) == {"experiment_id", "sample", "source_id"}
    assert set(sim.dparams.dims) == {"experiment_id", "sample", "source_id", "dparams"}

    # Get result for first experiment_id/source_id by setting other weights to zero
    sim.weights[:] = 0
    sim.weights.isel(experiment_id=0, source_id=0)[:] = 1 / len(sim.weights.sample)
    mix = XMixtureDistribution(params=sim.dparams, weights=sim.weights)
    out1 = getattr(mix, "sf")(0.5)

    # Get result for first experiment_id/source_id by selecting params
    out2 = xc.indices.stats.dist_method(
        function="sf",
        fit_params=sim.dparams.isel(experiment_id=0, source_id=0),
        arg=0.5,
    ).mean()
    assert np.isclose(out1.item(), out2.item())


@pytest.mark.parametrize("synthetic_jp_ds", ["wl_prcond"], indirect=True)
def test_IndicatorSimPRCOND(synthetic_jp_ds, synthetic_ds_daily):
    fut_period = (2070, 2100)
    wl_pot_backend, pr_cond_backend = synthetic_jp_ds
    pr_sim = synthetic_ds_daily
    obs = IndicatorObsPRCOND(data=pr_cond_backend, period=(1995, 2014))
    sim = IndicatorSimPRCOND(
        obs=obs, data=pr_sim, wl_pot=wl_pot_backend, period=fut_period
    )

    ### - TEST BASIC
    assert sim.long_name == "Precipitation extremes conditional on water level extremes"
    assert np.isclose(sim.sf(sim.isf(0.1)), 0.1, atol=1e-1)
    assert sim.isf(0.1) > obs.isf(0.1)
    assert sim.isf(0.5).name == "isf"

    ### - TEST FOR CONDITIONAL EXTREMES (i.e., no scale_pareto)
    # Check that likelihoods are not affected by peaks_per_yr
    assert not hasattr(sim, "peaks_per_yr")
    sim.peaks_per_yr = 4
    y1 = sim.sf(10)
    z1 = sim.pdf(10)
    sim.peaks_per_yr = 8
    y2 = sim.sf(10)
    z2 = sim.pdf(10)
    assert y2 == y1
    assert z2 == z1

    # Check that a change in period updates params.
    sim = IndicatorSimPRCOND(
        obs=obs, data=pr_sim, wl_pot=wl_pot_backend, period=fut_period
    )
    dparams1 = sim.dparams
    sim.period = (2020, 2050)
    dparams2 = sim.dparams
    assert not np.array_equal(dparams1.values, dparams2.values)

    #### - TEST FOR PRECIPITATION DELTA
    sim = IndicatorSimPRCOND(
        obs=obs, data=pr_sim, wl_pot=wl_pot_backend, period=fut_period
    )
    ref = (
        pr_sim.sel(
            experiment_id="ssp585",
            source_id="HadGEM3-GC31-MM",
            variant_label="r1i1p1f1",
        )
        .sel(time=slice("1995-01-01", "2014-12-31"))
        .quantile(0.95, dim="time")
        .item()
    )
    fut = (
        pr_sim.sel(
            experiment_id="ssp585",
            source_id="HadGEM3-GC31-MM",
            variant_label="r1i1p1f1",
        )
        .sel(time=slice("2070-01-01", "2100-12-31"))
        .quantile(0.95, dim="time")
        .item()
    )
    p95_delta = fut / ref
    pcond_delta = (
        sim._sample(period=(2060, 2090))
        .sel(
            experiment_id="ssp585",
            source_id="HadGEM3-GC31-MM",
            variant_label="r1i1p1f1",
        )
        .item()
    )
    assert (sim._sample(period=(2070, 2100)).median().item() - 1) > 0
    assert pcond_delta < p95_delta + 0.1
    assert (
        obs.dparams.sel(dparams="loc")
        < sim.fit(dist="norm", period=(2060, 2090)).sel(dparams="loc").median()
    )

    #### - TEST FOR WEIGHTS
    assert np.isclose(sim.weights.sum().item(), 1)
    assert set(sim.weights.dims) == {"experiment_id", "sample", "source_id"}
    assert set(sim.dparams.dims) == {"experiment_id", "sample", "source_id", "dparams"}

    # Get result for first experiment_id/source_id by setting other weights to zero
    sim.weights[:] = 0
    sim.weights.isel(experiment_id=0, source_id=0)[:] = 1 / len(sim.weights.sample)
    mix = XMixtureDistribution(params=sim.dparams, weights=sim.weights)
    out1 = getattr(mix, "sf")(0.5)

    # Get result for first experiment_id/source_id by selecting params
    out2 = xc.indices.stats.dist_method(
        function="sf",
        fit_params=sim.dparams.isel(experiment_id=0, source_id=0),
        arg=0.5,
    ).mean()
    assert np.isclose(out1.item(), out2.item())


@pytest.mark.parametrize("synthetic_jp_ds", ["pr_wlcond"], indirect=True)
def test_JPObs_pr_wlcond(synthetic_jp_ds):
    pr_pot_backend, wl_cond_backend = synthetic_jp_ds
    pr_pot = IndicatorObsPRPOT(data=pr_pot_backend, period=(1960, 2010))
    wl_cond = IndicatorObsWLCOND(data=wl_cond_backend, period=(1960, 2010))
    jp = IndicatorObsJP(
        obs_pot=pr_pot, obs_cond=wl_cond, name="jp", period=(1960, 2010)
    )
    assert (
        jp.long_name
        == "Joint precipitation peaks over threshold and conditional water extremes"
    )
    assert jp.sf([3, 3]).name == "sf"

    # Check that the marginals are properly assigned
    assert jp.obs_pot.data.name == "pr_pot"
    assert jp.obs_cond.data.name == "wl_cond"

    # Check the pseudo-observations
    assert (jp.data > 0).all() and (jp.data < 1).all()

    # Check bic is lower for the expected distribution
    assert jp._bic(period=(1960, 2010), dist="clayton") < jp._bic(
        period=(1960, 2010), dist="gaussian"
    )

    # Check that the expected distribution was selected
    assert jp.dist == "clayton"
    assert jp._bic(period=jp.period, dist=jp.dist) == jp.bic

    # Check that a change in metric triggers a change in metrics
    jp.metric = "aic"
    jp._update_metrics()  # TODO - why does this not trigger with @param.depends("metric", watch=True)?
    jp._update_params()  # TODO - why does this not trigger with @param.depends("metrics", watch=True)?
    assert jp.metrics_da.attrs.get("long_name") == "Akaike Information Criterion"

    # The likelihood of upper tail values should increase as the number of peaks per year increases
    jp.obs_pot.peaks_per_yr = 4
    y1 = jp.sf([3, 3])
    jp.obs_pot.peaks_per_yr = 8
    y2 = jp.sf([3, 3])
    assert y2 == y1 * 2

    # Check that student copula has two parameters
    jp.dist = "student"
    jp._update_params()  # TODO - why does this not trigger with @param.depends@param.depends("dist", watch=True)?
    assert len(jp.dparams) == 2

    # Test that inf bic does not impeed distribution selection
    metrics_da = jp.metrics_da.where(
        jp.metrics_da.scipy_dist != "student", other=np.inf
    )
    assert metrics_da.idxmin("scipy_dist").item() == "clayton"

    # Ensure only water level data is affected by a period change (renormalized)
    pr_dparams1, wl_dparams1 = jp.obs_pot.dparams.values, jp.obs_cond.dparams.values
    pr_len1, wl_len1 = len(jp.obs_pot.data), len(jp.obs_cond.data)
    pobs1 = jp.data.values
    jp.period = (1995, 2014)
    pr_dparams2, wl_dparams2 = jp.obs_pot.dparams.values, jp.obs_cond.dparams.values
    pr_len2, wl_len2 = len(jp.obs_pot.data), len(jp.obs_cond.data)
    pobs2 = jp.data.values

    assert np.array_equal(pr_dparams1, pr_dparams2)
    assert not np.array_equal(wl_dparams1, wl_dparams2)
    assert pr_len1 == pr_len2 & wl_len1 == wl_len2
    assert np.array_equal(pobs1, pobs2)

    # Test dist method
    p1 = 1 - jp.obs_pot.sf(3.1).item() / jp.obs_pot.peaks_per_yr
    p2 = 1 - jp.obs_cond.sf(3).item()
    unscaled_joint_sf = jp.sf([3.1, 3]) / jp.obs_pot.peaks_per_yr
    unscaled_joint_cdf = jp.cdf([3.1, 3]) / jp.obs_pot.peaks_per_yr
    assert np.isclose(
        unscaled_joint_sf.values, (1 - p1 - p2 + unscaled_joint_cdf).values
    )


@pytest.mark.parametrize("synthetic_jp_ds", ["wl_prcond"], indirect=True)
def test_JPObs_wl_prcond(synthetic_jp_ds):
    wl_pot_backend, pr_cond_backend = synthetic_jp_ds
    wl_pot = IndicatorObsWL(data=wl_pot_backend, period=(1960, 2010))
    pr_cond = IndicatorObsPRCOND(data=pr_cond_backend, period=(1960, 2010))
    jp = IndicatorObsJP(
        obs_pot=wl_pot,
        obs_cond=pr_cond,
        name="wl_prcond",
        period=(1960, 2010),
        dist="clayton",
    )
    assert (
        jp.long_name
        == "Joint water level peaks over threshold and conditional precipitation extremes"
    )
    assert jp.sf([3, 3]).name == "sf"

    # Check that the marginals are properly assigned
    assert jp.obs_pot.data.name == "wl_pot"
    assert jp.obs_cond.data.name == "pr_cond"

    # Ensure only water level data is affected by a period change (renormalized)
    wl_dparams1, pr_dparams1 = jp.obs_pot.dparams.values, jp.obs_cond.dparams.values
    wl_len1, pr_len1 = len(jp.obs_pot.data), len(jp.obs_cond.data)
    pobs1 = jp.data.values
    jp_dparams1 = jp.dparams
    jp.period = (1995, 2014)
    wl_dparams2, pr_dparams2 = jp.obs_pot.dparams.values, jp.obs_cond.dparams.values
    wl_len2, pr_len2 = len(jp.obs_pot.data), len(jp.obs_cond.data)
    pobs2 = jp.data.values
    jp_dparams2 = jp.dparams

    assert np.array_equal(pr_dparams1, pr_dparams2)
    assert not np.array_equal(wl_dparams1, wl_dparams2)
    assert pr_len1 == pr_len2 & wl_len1 == wl_len2
    assert np.array_equal(pobs1, pobs2)
    assert jp_dparams1.equals(jp_dparams2)

    # Test dist method
    p1 = 1 - jp.obs_pot.sf(3.1).item() / jp.obs_pot.peaks_per_yr
    p2 = 1 - jp.obs_cond.sf(3).item()
    unscaled_joint_sf = jp.sf([3.1, 3]) / jp.obs_pot.peaks_per_yr
    unscaled_joint_cdf = jp.cdf([3.1, 3]) / jp.obs_pot.peaks_per_yr
    assert np.isclose(
        unscaled_joint_sf.values, (1 - p1 - p2 + unscaled_joint_cdf).values
    )


@pytest.mark.parametrize("synthetic_jp_ds", ["pr_wlcond"], indirect=True)
def test_JPSim_pr_wlcond(synthetic_jp_ds, synthetic_ds_daily, synthetic_ewl_ds):
    pr_pot_backend, wl_cond_backend = synthetic_jp_ds
    _, _, sl, _ = synthetic_ewl_ds
    pr_sim = synthetic_ds_daily

    pr_obs = IndicatorObsPRPOT(data=pr_pot_backend, period=(1960, 2010))
    wl_obs = IndicatorObsWLCOND(data=wl_cond_backend, period=(1960, 2010))
    cop_obs = IndicatorObsJP(
        obs_pot=pr_obs, obs_cond=wl_obs, name="pr_wlcond", period=(1960, 2010)
    )
    pr_sim = IndicatorSimPRPOT(obs=pr_obs, data=pr_sim, period=(2020, 2050))
    wl_sim = IndicatorSimWLCOND(obs=wl_obs, data=sl, period=(2020, 2050))

    jp = IndicatorSimJP(
        sim_pot=pr_sim, sim_cond=wl_sim, obs_cop=cop_obs, period=(2020, 2050)
    )
    assert (
        jp.long_name
        == "Joint precipitation peaks over threshold and conditional water extremes"
    )
    assert jp.sim_pot.obs.data.name == "pr_pot"
    assert jp.sim_cond.obs.data.name == "wl_cond"
    assert (jp.data > 0).all() and (jp.data < 1).all()
    assert jp.dist == "clayton"
    assert jp.sf([3, 3]) > cop_obs.sf([3, 3])
    assert jp.sf([3, 3]).name == "sf"

    # The likelihood of upper tail values should increase as the number of peaks per year increases
    jp.sim_pot.obs.peaks_per_yr = 4
    y1 = jp.sf([3, 3])
    jp.sim_pot.obs.peaks_per_yr = 8
    y2 = jp.sf([3, 3])
    assert y2 == y1 * 2

    # Ensure that a change in period affects marginal dparams but not the pseudo-observations or copula dparams.
    pr_dparams1, wl_dparams1 = jp.sim_pot.dparams.values, jp.sim_cond.dparams.values
    pobs1 = jp.data.values
    jp_dparams1 = jp.dparams
    jp.period = (2070, 2100)
    pr_dparams2, wl_dparams2 = jp.sim_pot.dparams.values, jp.sim_cond.dparams.values
    pobs2 = jp.data.values
    jp_dparams2 = jp.dparams

    assert not np.array_equal(pr_dparams1, pr_dparams2)
    assert not np.array_equal(wl_dparams1, wl_dparams2)
    assert np.array_equal(pobs1, pobs2)
    assert jp_dparams1.equals(jp_dparams2)

    # Test dist method
    p1 = 1 - jp.sim_pot.sf(3.1).item() / jp.sim_pot.obs.peaks_per_yr
    p2 = 1 - jp.sim_cond.sf(3).item()
    unscaled_joint_sf = jp.sf([3.1, 3]) / jp.sim_pot.obs.peaks_per_yr
    unscaled_joint_cdf = jp.cdf([3.1, 3]) / jp.sim_pot.obs.peaks_per_yr
    assert np.isclose(
        unscaled_joint_sf.values, (1 - p1 - p2 + unscaled_joint_cdf).values
    )


@pytest.mark.parametrize("synthetic_jp_ds", ["wl_prcond"], indirect=True)
def test_JPSim_wl_prcond(synthetic_jp_ds, synthetic_ds_daily, synthetic_ewl_ds):
    wl_pot_backend, pr_cond_backend = synthetic_jp_ds
    _, _, sl, _ = synthetic_ewl_ds
    pr_sim = synthetic_ds_daily

    wl_obs = IndicatorObsWL(data=wl_pot_backend, period=(1960, 2010))
    pr_obs = IndicatorObsPRCOND(data=pr_cond_backend, period=(1960, 2010))
    cop_obs = IndicatorObsJP(
        obs_pot=wl_obs, obs_cond=pr_obs, name="pr_wlcond", period=(1960, 2010)
    )
    pr_sim = IndicatorSimPRCOND(
        obs=pr_obs, data=pr_sim, wl_pot=wl_pot_backend, period=(2020, 2050)
    )
    wl_sim = IndicatorSimWL(obs=wl_obs, data=sl, period=(2020, 2050))

    jp = IndicatorSimJP(
        sim_pot=wl_sim, sim_cond=pr_sim, obs_cop=cop_obs, period=(2020, 2050)
    )
    assert (
        jp.long_name
        == "Joint water level peaks over threshold and conditional precipitation extremes"
    )
    assert jp.sim_pot.obs.data.name == "wl_pot"
    assert jp.sim_cond.obs.data.name == "pr_cond"
    assert (jp.data > 0).all() and (jp.data < 1).all()
    assert jp.dist == "clayton"
    assert jp.sf([3, 3]) > cop_obs.sf([3, 3])
    assert jp.sf([3, 3]).name == "sf"

    # The likelihood of upper tail values should increase as the number of peaks per year increases
    jp.sim_pot.obs.peaks_per_yr = 4
    y1 = jp.sf([3, 3])
    jp.sim_pot.obs.peaks_per_yr = 8
    y2 = jp.sf([3, 3])
    assert y2 == y1 * 2

    # Ensure that a change in period affects marginal dparams but not the pseudo-observations or copula dparams.
    wl_dparams1, pr_dparams1 = jp.sim_pot.dparams.values, jp.sim_cond.dparams.values
    pobs1 = jp.data.values
    jp_dparams1 = jp.dparams
    jp.period = (2070, 2100)
    wl_dparams2, pr_dparams2 = jp.sim_pot.dparams.values, jp.sim_cond.dparams.values
    pobs2 = jp.data.values
    jp_dparams2 = jp.dparams

    assert not np.array_equal(pr_dparams1, pr_dparams2)
    assert not np.array_equal(wl_dparams1, wl_dparams2)
    assert np.array_equal(pobs1, pobs2)
    assert jp_dparams1.equals(jp_dparams2)

    # Test dist method
    p1 = 1 - jp.sim_pot.sf(3.1).item() / jp.sim_pot.obs.peaks_per_yr
    p2 = 1 - jp.sim_cond.sf(3).item()
    unscaled_joint_sf = jp.sf([3.1, 3]) / jp.sim_pot.obs.peaks_per_yr
    unscaled_joint_cdf = jp.cdf([3.1, 3]) / jp.sim_pot.obs.peaks_per_yr
    assert np.isclose(
        unscaled_joint_sf.values, (1 - p1 - p2 + unscaled_joint_cdf).values
    )
