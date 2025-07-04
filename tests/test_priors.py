import numpy as np

from portail_ing.risk.priors import (
    members,
    model_weights_from_sherwood,
    scenario_weights_from_iams,
    weights,
)


def test_model_weights_from_sherwood():
    w = model_weights_from_sherwood(["INM-CM4-8", "GFDL-CM4", "HadGEM3-GC31-MM"])
    assert "source_id" in w.dims
    assert len(w) == 3
    np.testing.assert_almost_equal(w.sum(), 1)


def test_scenario_weights_from_iams():
    w = scenario_weights_from_iams()
    assert "experiment_id" in w.dims
    assert "time" in w.dims

    mw = w.sel(time=(slice("2050", "2070"))).mean(dim="time")
    assert mw.sum() == 1


def test_members(synthetic_dataset_fut):
    ds = synthetic_dataset_fut
    da = ds["00"].unstack("realization")
    w = members(da)
    assert "source_id" in w.dims
    assert "experiment_id" in w.dims
    assert w.isel(source_id=0, experiment_id=0) == 1
    assert w.isel(source_id=1, experiment_id=0) == 0.5


def test_weights(synthetic_dataset_fut):
    ds = synthetic_dataset_fut
    w = weights(ds["00"])
    np.testing.assert_almost_equal(w.sum(), 1)
