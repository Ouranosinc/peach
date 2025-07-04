import os

import pandas as pd
import pytest

import portail_ing.frontend.parameters as p
import portail_ing.frontend.views as v

# For now these tests are just smoke tests
# TODO: Save the plots and compare them to reference images


def test_station_and_indicator_list_viewer(station_data, config):

    # Parameters

    gl = p.Global(locale="fr")

    site = p.Site(lat=44.5, lon=-65.9)
    map_param = p.Map(clat=44.5, clon=-65.9, z=7)

    s = p.Station(station_data, variables=["pr", "tas"], site=site, locale=gl.param.locale)
    il = p.IndicatorList(
        config=config[0], locale=gl.param.locale, station_id=s.param.station_id
    )

    # Indicator list viewer
    v.IndicatorListViewer(inds=il, config=config[2]).__panel__()

    # Station viewer
    v.StationViewer(
        station=s, site=site, map_param=map_param, config=config[1]
    ).__panel__()


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skipping this test on GitHub CI"
)
def test_analysis_viewer(
    hdd_series, idf_obs, idf_sim, wl_pot_obs, wl_pot_sim, station_data
):
    obs, sim = hdd_series
    links = {
        "obs": {"00": obs, "11": idf_obs, "22": wl_pot_obs},
        "sim": {"00": sim, "11": idf_sim, "22": wl_pot_sim},
    }

    df = pd.concat(
        [
            station_data.query("station == '8202251'"),
            station_data.query("station == '00490'"),
        ]
    )
    a = p.Analysis(station_df=df)
    a._load_results(links=links)

    # Reference analysis viewer
    v.ObsAnalysisViewer(analysis=a).__panel__()

    # Future analysis viewer
    v.FutAnalysisViewer(analysis=a).__panel__()


def test_analysis_WL_viewer(synthetic_ewl_ds, station_data):
    wl, wl_pot, sl, stn_thresh = synthetic_ewl_ds

    # import xarray as xr
    # wl_pot = xr.open_dataset("src/portail_ing/risk/cbcl_workflow/test_data/00490_wl_pot.nc").wl_pot
    # sl = xr.open_dataset("src/portail_ing/risk/cbcl_workflow/test_data/00490_sl.nc").sl_delta
    #
    # wl_pot.attrs["long_name"] = "relative sea level"
    # wl_pot = wl_pot.rename(time_ref='time')
    # wl_pot = wl_pot.isel(realization=0, period=0, drop=True)

    a = p.Analysis(
        ds={"obs": {"123": wl_pot}, "sim": {"123": sl}},
    )

    # Reference analysis viewer
    v.ObsAnalysisViewer(analysis=a).__panel__()

    # Future analysis viewer
    v.FutAnalysisViewer(analysis=a).__panel__()

    v.IndicatorDAViewer(ind=a.obs["123"]).__panel__()
    v.IndicatorSimDAViewer(ind=a.ref["123"]).__panel__()
    v.IndicatorSimDAViewer(ind=a.fut["123"]).__panel__()


def test_hazard_matrix_viewer(synthetic_dataset, synthetic_dataset_fut, station_data):

    sim_ds = synthetic_dataset_fut

    for key, val in sim_ds.items():
        val.data += 0.01

    a = p.Analysis(
        ds={"obs": synthetic_dataset, "sim": synthetic_dataset_fut},
        level=0.01,
    )
    hm = p.HazardMatrix(analysis=a)
    mv = v.HazardMatrixViewer(hm=hm)
    mv.__panel__()


def test_application(
    station_data, config, synthetic_dataset, synthetic_dataset_fut, tmp_path
):

    gl = p.Global(locale="fr")

    site = p.Site(
        locale=gl.param.locale,
    )

    map_param = p.Map(
        sync_url=False,
        locale=gl.param.locale,
    )
    # Station viewer
    s = p.Station(station_data, variables=["pr", "tas"], site=site, locale=gl.param.locale)

    # Indicator list viewer
    il = p.IndicatorList(
        config=config[0], locale=gl.param.locale, station_id=s.param.station_id
    )

    # Initialization
    s.pr = "7028441"
    il.add("WETDAYS")
    il.add("DRY_DAYS")

    # Analysis
    a = p.Analysis(
        ds={"obs": synthetic_dataset, "sim": synthetic_dataset_fut},
    )

    # Hazard matrix
    hm = p.HazardMatrix(analysis=a)

    # Application
    app = v.Application(
        global_=gl,
        site=site,
        map_param=map_param,
        station=s,
        indicators=il,
        analysis=a,
        hazmat=hm,
        config=config[3],
    )
    app.__panel__()

    app.export_results()
    app.export_request()
