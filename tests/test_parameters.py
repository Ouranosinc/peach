# ruff: noqa: D103
"""Tests for peach.src.parameters."""
import os
import time

import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr

import peach.frontend as fe
import peach.frontend.parameters as p


def test_IndexingIndicatorArguments():
    fs = p.IndexingIndicatorArguments()

    assert fs.freq == "YS-JAN"
    assert fs.indexer == {}

    fs.start_m = 6
    assert fs.freq == "YS-JUN"
    assert "month" in fs.indexer
    assert all(fs.indexer["month"] == np.array([6, 7, 8, 9, 10, 11, 12]))

    # test translation
    assert fs.locale == "fr"
    assert fs.param.start_m.label == "Mois de début"
    fs.locale = "en"
    assert fs.param.start_m.label == "Start month"

    assert tuple(fs.param.end_m.objects) == tuple(range(1, 13))


def test_Station(station_data):
    variables = ["pr", "tas"]
    site = p.Site(locale="fr")
    s = p.Station(station_data, variables=variables, site=site, locale="fr")

    # Check that attributes for variables have been created
    for v in variables:
        assert hasattr(s, v)

    # Set a variable attribute
    s.pr = "7028441"
    assert s.station_id["pr"] == "7028441"

    # Check that setting tas sets tasmin and tasmax
    s.tas = "7028442"
    assert s.station_id["tasmax"] == "7028442"
    assert s.station_id["tasmin"] == "7028442"

    assert type(s.df) == pd.DataFrame


def test_Indicator():
    iid = "HEATING_DEGREE_DAYS"
    config = {"args": {"thresh": {"default_units": "degC", "vmin": 10, "vmax": 30}}}
    ind = p.GenericIndicator.from_xclim(iid, config=config, locale="en")

    # Test translation
    assert ind.title == "Heating degree days"

    ind.locale = "fr"
    assert ind.title == "Degrés-jours de chauffage"

    # Test variable logic
    assert ind.variables == []
    assert ind.param.variables.objects == [
        "tas",
    ]
    assert not ind.has_data

    with pytest.raises(NotImplementedError):
        ind.to_dict()

    # Specify station data
    ind.station_id = {"pr": 6102857, "tas": None}
    assert not ind.has_data

    ind.station_id = {"tas": 6104027}
    assert ind.has_data
    assert ind.variables == ["tas"]

    indC = p.IndicatorComputation.from_generic(ind)
    assert isinstance(indC.to_dict(), dict)
    assert isinstance(indC.hash, str)
    assert "thresh" in indC.args.param
    assert "thresh_unit" in indC.args.param
    assert indC.args.param.start_m.label == "Mois de début"
    assert indC.args.param.thresh.softbounds == (10, 30)
    assert indC.base.variables == ["tas"]

    assert len(indC.uuid) == 36


def test_IndicatorBase_IDF():

    iid = "IDF"
    config = {
        "args": {
            "duration": {
                "choices": ["1h", "2h", "6h", "12h", "24h"],
                "label": {"fr": "Durée"},
                "doc": {"fr": "Durée de la pluie"},
            }
        }
    }

    ind = p.GenericIndicator.from_xclim(iid, config=config, locale="en")
    indC = p.IndicatorComputation.from_generic(ind)
    assert "duration" in indC.args.param

    indC.base.station_id = {"tas": "000", "idf": "111"}
    indC.args.duration = "2h"
    assert "duration" in indC.to_dict()["params"]
    indC.to_dict()["params"]["duration"] == "2h"


def test_IndicatorList():
    config = {
        "HEATING_DEGREE_DAYS": {
            "args": {"thresh": {"default_units": "degC", "vmin": 10, "vmax": 30}}
        }
    }

    iid = "HEATING_DEGREE_DAYS"
    indl = p.IndicatorList(config=config, locale="fr", station_id={"tas": "7028442"})
    assert indl.indicators[iid].has_data

    # Make sure we can have multiple copies of the same indicator
    uuid_1 = indl.add(iid)
    assert len(indl.selected) == 1
    uuid_2 = indl.add(iid)
    assert len(indl.selected) == 2

    indl.remove(uuid_1)
    assert len(indl.selected) == 1
    assert uuid_2 in indl.selected

    assert "thresh" in indl.selected[uuid_2].args.param


@pytest.mark.backend
def test_IndicatorList_compute():
    try:
        requests.get("http://localhost:8081/api/openapi")
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running")

    config = {
        "WETDAYS": {
            "args": {"thresh": {"default_units": "mm/d", "vmin": 0, "vmax": 10}}
        }
    }

    iid = "WETDAYS"
    indl = p.IndicatorList(
        config=config,
        locale="fr",
        station_id={"pr": "7028441"},
        backend="http://localhost:8081/api/",
    )
    uuid = indl.add(iid)
    indl.post_all_requests()
    assert indl.selected[uuid].obs_job.state > 0
    assert indl.selected[uuid].sim_job.state > 0


def test_analysis_hdd(hdd_series, station_data):
    obs, sim = hdd_series

    a = p.Analysis(level=0.01)
    a._load_results(links={"obs": {"00": obs}, "sim": {"00": sim}})
    assert a.param.ref_period.bounds == (1950, 2020)
    assert a.param.fut_period.bounds == (1960, 2100)

    # Test translation
    a.locale = "fr"
    assert a.param.ref_period.label == "Période de référence"


@pytest.mark.online
@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping this test on GitHub CI"
)
def test_analysis_type_map(idf_obs, idf_sim, wl_pot_obs, wl_pot_sim):
    """Make sure analysis creates the right type of Indicator instance."""
    obs = {
        "IDF": idf_obs,
        "WL_POT": wl_pot_obs,
    }
    sim = {
        "IDF": idf_sim,
        "WL_POT": wl_pot_sim,
    }

    a = p.Analysis(level=0.01)
    a._load_results(links={"obs": obs, "sim": sim})

    assert isinstance(a.obs["IDF"], fe.idf_parameters.IndicatorObsIDF)
    assert isinstance(a.ref["IDF"], fe.idf_parameters.IndicatorRefIDF)
    assert isinstance(a.fut["IDF"], fe.idf_parameters.IndicatorSimIDF)

    assert isinstance(a.obs["WL_POT"], fe.wl_parameters.IndicatorObsWL)
    assert isinstance(a.ref["WL_POT"], fe.wl_parameters.IndicatorRefWL)
    assert isinstance(a.fut["WL_POT"], fe.wl_parameters.IndicatorSimWL)


def test_IndicatorDA(synthetic_dataset):
    ds = synthetic_dataset

    ida = p.IndicatorDA.from_da(da=ds["00"], period=(1970, 2010))
    assert len(ida.sample) == 41
    assert ida.dparams is None

    # Test cache - This is not discriminating enough.
    # now = time.perf_counter()
    # p1 = ida.fit("t", ida.period)
    # tic = time.perf_counter()
    # p2 = ida.fit("t", ida.period)
    # tac = time.perf_counter()
    # assert (tac - tic) < (tic - now), f"Caching is not working as expected : 1st: {(tic - now) * 1000:.0f} ms, 2nd : {(tac - tic) * 1000:.0f} ms"

    # xr.testing.assert_equal(p1, p2)
    # assert "dparams" in p2.dims

    # Check that changing distribution updates the params
    ida.dist = "norm"
    assert isinstance(ida.dparams, xr.DataArray)
    assert len(ida.dparams) == 2
    pu = tuple(ida.dparams.data)

    # Check that changing the period updates the params
    # Ureliable if dist is uniform, because the parameters depend only on the sample min/max
    ida.period = (1979, 2020)
    assert len(ida.sample) == 42
    time.sleep(0.01)
    pp = tuple(ida.dparams.data)
    assert isinstance(ida.dparams, xr.DataArray)
    assert pp != pu

    # Check pdf
    ida.dist = "uniform"
    y = ida.pdf(xr.DataArray([0.3, 0.4, 0.5], dims=("xx")))
    np.testing.assert_allclose(y, 1, rtol=0.2)

    np.testing.assert_almost_equal(ida.sf(0.5), 0.5, decimal=1)
    np.testing.assert_almost_equal(ida.isf(0.5), 0.5, decimal=1)


def test_IndicatorObsDA(synthetic_dataset, station_data):
    ds = synthetic_dataset

    ida = p.IndicatorObsDA(
        data=ds["00"], name="Var_0", period=(1960, 2010), dist="norm"
    )

    assert isinstance(ida.dparams, xr.DataArray)
    assert len(ida.dparams) == 2

    assert isinstance(ida.bic, xr.DataArray)
    assert "scipy_dist" in ida.bic.attrs
    assert ida.bic.attrs["scipy_dist"] == "norm"

    assert ida.best_dist() == "uniform"
    ida.metric = "aic"
    assert ida.best_dist() == "uniform"

    # Test automatic distribution selection
    ida = p.IndicatorObsDA(data=ds["00"], name="Var_0", period=(1960, 2010))
    assert ida.dist == "uniform"

    assert isinstance(ida.ts_caption, str)


def test_IndicatorRefDA(synthetic_dataset_fut, synthetic_dataset):
    sim_ds = synthetic_dataset_fut
    obs_ds = synthetic_dataset

    o = p.IndicatorObsDA.from_da(da=obs_ds["00"], period=(1960, 2010), dist="uniform")
    o.long_name == "Var en 0"

    r = p.IndicatorRefDA(data=sim_ds["00"], dist=o.param.dist, obs=o, level=0.01)

    o.dist = "norm"
    assert r.dist == "norm"

    r.dist = "uniform"
    assert o.dist == "norm"

    assert isinstance(r.dparams, xr.DataArray)
    assert {"source_id", "experiment_id", "dparams"} == set(r.dparams.dims)
    assert {"source_id", "experiment_id"} == set(r.weights.dims)
    np.testing.assert_array_almost_equal(r.weights.sum(), 1)

    y = r.pdf([0.35, 0.4, 0.9])
    np.testing.assert_allclose(y, 1, rtol=0.2)

    np.testing.assert_almost_equal(r.sf(1), 0.1, decimal=1)
    np.testing.assert_almost_equal(r.isf(0.1), 1.0, decimal=1)

    per = r.experiment_percentiles([1, 50, 90])
    assert set(per.dims) == {"experiment_id", "time"}

    # Test weights
    r = p.IndicatorRefDA(data=sim_ds["00"], dist="uniform", obs=o, level=0.005)
    assert len(r._ks) == 5
    assert r._ks.all()

    # Should fail due to not enough acceptable models
    with pytest.raises(ValueError):
        p.IndicatorRefDA(data=sim_ds["00"] + 0.5, obs=o)

    # Test model weights with one bad model
    bdata = sim_ds["00"].copy()
    bdata.loc[{"source_id": "INM-CM4-8"}] = 1
    fi = p.IndicatorRefDA(data=bdata, obs=o, level=0.005)
    assert not fi._ks.sel(source_id="INM-CM4-8")
    assert fi._ks.sum() == 4
    assert fi.model_weights.sel(source_id="INM-CM4-8").values == 0


def test_HazardThreshold(synthetic_dataset, synthetic_dataset_fut):

    obs_da = synthetic_dataset["00"]
    sim_da = synthetic_dataset_fut["00"]

    obs_ind = p.IndicatorObsDA(data=obs_da, period=(1960, 2010), dist="uniform")
    ref_ind = p.IndicatorRefDA(data=sim_da, dist=obs_ind.dist, obs=obs_ind, level=0.005)
    fut_ind = p.IndicatorSimDA(
        data=sim_da, period=(2021, 2050), dist=obs_ind.dist, weights=ref_ind.weights
    )

    ht = p.HazardThreshold(obs=obs_ind, ref=ref_ind, fut=fut_ind, input="X", value=0.9)

    np.testing.assert_almost_equal(ht.obs_sf, 0.1, 1)
    np.testing.assert_almost_equal(ht.obs_t, 10, -1)
    np.testing.assert_almost_equal(ht.ref_sf, 0.1, 1)
    np.testing.assert_almost_equal(ht.fut_sf, 0.2, 1)
    np.testing.assert_almost_equal(ht.ratio, 1.2, 0)

    assert isinstance(ht.values, pd.Series)
    assert isinstance(ht.titles, dict)


def test_HazardMatrix(synthetic_dataset, synthetic_dataset_fut, station_data):
    a = p.Analysis(
        ds={"obs": synthetic_dataset, "sim": synthetic_dataset_fut},
        level=0.01,
    )
    hm = p.HazardMatrix(analysis=a)

    assert len(hm.matrix["00"]) == 1
    hm.add("00", 0)
    assert len(hm.matrix["00"]) == 2
    hm.remove("00", 1)
    assert len(hm.matrix["00"]) == 1

    assert isinstance(hm.df, pd.DataFrame)
