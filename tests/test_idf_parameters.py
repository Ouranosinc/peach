import os

import pytest
import xarray as xr

from portail_ing.frontend.idf_parameters import (
    IndicatorObsIDF,
    IndicatorRefIDF,
    IndicatorSimIDF,
)
from portail_ing.frontend.parameters import Analysis, HazardMatrix


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping this test on GitHub CI"
)
def test_IndicatorsIDF(idf_obs, idf_sim):

    # Observations

    dao = xr.open_dataarray(idf_obs, engine="zarr")
    o = IndicatorObsIDF.from_da(da=dao, period=(1970, 2010))

    assert o.duration == "1h"
    assert o.station_id == "7027725"
    assert o.data.attrs["units"] == "m/h"
    assert o.sample.attrs["units"] == "mm"
    assert o.ts.attrs["units"] == "mm"

    # Simulations
    das = xr.open_dataarray(idf_sim, engine="zarr")

    ref = IndicatorRefIDF.from_da(da=das, obs=o)
    assert ref._ks.all()

    fut = IndicatorSimIDF.from_da(
        da=das, obs=o, period=(2040, 2070), model_weights=ref.param.model_weights
    )
    assert {"t", "time", "variant_label", "source_id", "experiment_id"}.issubset(
        fut.ts.dims
    )

    # Test that delta returns a DataArray without time dimension
    d = fut.delta(fut.period)
    assert "time" not in d.dims
    assert d.attrs["units"] == "°C"

    per = fut.experiment_percentiles(per=[10, 50, 90])
    assert "time" in per.dims
    assert not per.idf_p50.isnull().all()

    # Test analysis
    a = Analysis(ds={"obs": {"123": dao}, "sim": {"123": das}})
    hm = HazardMatrix(analysis=a)
    assert len(hm.matrix) == 1
