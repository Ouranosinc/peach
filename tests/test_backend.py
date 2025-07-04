import os

import pytest
import xarray as xr

# This is just testing the import. The backend computations are included in some of the pytest fixtures.


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping this test on GitHub CI"
)
def test_translation(idf_obs, idf_sim):
    # idf_sim is fairly long to run, do not abandon hope
    da = xr.open_dataarray(idf_obs, engine="zarr")
    assert "long_name_fr" in da.attrs
    assert "description_fr" in da.attrs

    da = xr.open_dataarray(idf_sim, engine="zarr")
    assert "long_name_fr" in da.attrs
    assert "description_fr" in da.attrs


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping this test on GitHub CI"
)
def test_hdd(hdd_series):
    obs, sim = hdd_series
    da = xr.open_dataarray(obs, engine="zarr")
    assert da.attrs["id"] == "HEATING_DEGREE_DAYS"


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping this test on GitHub CI"
)
def test_wl_pot(wl_pot_obs, wl_pot_sim):
    da = xr.open_dataarray(wl_pot_obs, engine="zarr")
    assert da.attrs["id"] == "WL_POT"
    assert "peaks_per_yr" in da.attrs
    assert "stn_thresh" in da.attrs

    da = xr.open_dataarray(wl_pot_sim, engine="zarr")
    assert da.attrs["id"] == "WL_POT"
    # assert da.attrs["long_name"] == "relative_sea_level"
    assert "sl_mm_yr" in da.attrs
