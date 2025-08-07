from pathlib import Path

import pytest
import yaml

from peach import test_utils as tu
from peach.common import config as ping_config

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "src" / "peach" / "frontend" / "config"


@pytest.fixture
def synthetic_dataset():
    return tu.synthetic_ds()


@pytest.fixture
def synthetic_dataset_fut():
    return tu.synthetic_ds_fut()


@pytest.fixture
def synthetic_ds_daily():
    return tu.synthetic_ds_daily()


@pytest.fixture
def station_data():
    return ping_config.read_stations()


@pytest.fixture(scope="session")
def tmp_workspace(tmp_path_factory):
    path = tmp_path_factory.mktemp("data")
    ping_config.__dict__["WORKSPACE"] = Path(path)
    return Path(path)


@pytest.fixture
def config():

    with open(CONFIG / "indicators.yml") as file:
        c1 = yaml.safe_load(file)
    with open(CONFIG / "station_select.yml") as file:
        c2 = yaml.safe_load(file)
    with open(CONFIG / "indicator_select.yml") as file:
        c3 = yaml.safe_load(file)
    with open(CONFIG / "application.yml") as file:
        c4 = yaml.safe_load(file)

    return c1, c2, c3, c4


@pytest.fixture
def synthetic_ewl_ds():
    return tu.synthetic_ewl_ds()


@pytest.fixture
def synthetic_ewl_ds_lea():
    wl, wl_pot, sl, stn_thresh = tu.synthetic_ewl_ds()

    wl = wl.rename({"time": "time_ref"})

    wl_pot = wl_pot.rename({"time": "time_ref"})
    wl_pot = wl_pot.expand_dims({"realization": ["obs"], "period": ["ref"]})

    sl = sl.isel(realization=0, drop=True)
    return wl, wl_pot, sl, stn_thresh


@pytest.fixture(scope="session")
def idf_obs(tmp_workspace):
    from peach.backend.compute_indicators import ComputeIDFProcessorOBS

    # Make IDF Obs
    p = ComputeIDFProcessorOBS({"name": "Test-IDF-Obs"})
    p.INPUT_DATASET_PATTERN = (
        "s3://https://minio.ouranos.ca/portail-ing/IDF3.30.zarr"
    )

    data = {
        "name": "IDF",
        "params": {"duration": "1h"},
        "stations": {"idf": "7027725"},
    }
    mimetype, output = p.execute(data)
    return tmp_workspace / output["value"]


@pytest.fixture(scope="session")
def idf_sim(tmp_workspace):
    from peach.backend.compute_indicators import ComputeIDFProcessorSIM

    # Make IDF Obs
    p = ComputeIDFProcessorSIM({"name": "Test-IDF-Sim"})
    p.INPUT_DATASET_PATTERN = "s3://https://minio.ouranos.ca/portail-ing/portail_ing_{var}_CMIP6_stations_AHCCD_concat.zarr"

    data = {
        "name": "IDF",
        "params": {"duration": "1h"},
        "stations": {"tas": "7033650"},
    }
    mimetype, output = p.execute(data)
    return tmp_workspace / output["value"]


@pytest.fixture(scope="session")
def wl_pot_obs(tmp_workspace):
    from peach.backend.compute_indicators import ComputeWaterLevelProcessorOBS

    # Make IDF Obs
    p = ComputeWaterLevelProcessorOBS({"name": "Test-WL-Obs"})
    p.INPUT_DATASET_PATTERN = "s3://https://minio.ouranos.ca/portail-ing/WL/{station_id}_{var}.nc"

    data = {
        "name": "WL_POT",
        "params": {},
        "stations": {"wl_pot": "00065"},
    }
    mimetype, output = p.execute(data)
    return tmp_workspace / output["value"]


@pytest.fixture(scope="session")
def wl_pot_sim(tmp_workspace):
    from peach.backend.compute_indicators import ComputeWaterLevelProcessorSIM

    # Make IDF Obs
    p = ComputeWaterLevelProcessorSIM({"name": "Test-WL-Sim"})
    p.INPUT_DATASET_PATTERN = "s3://https://minio.ouranos.ca/portail-ing/SL/{station_id}_{var}.nc"

    data = {
        "name": "WL_POT",
        "params": {},
        "stations": {"wl_pot": "00065"},
    }
    mimetype, output = p.execute(data)
    return tmp_workspace / output["value"]


@pytest.fixture(scope="session")
def tas_obs(tmp_path_factory):
    ds = tu.tas_obs().to_dataset()
    path = tmp_path_factory.mktemp("data") / "tas_obs.zarr"
    ds.to_zarr(path)
    return path
    # store = tu.MemoryZarrStore()
    # ds.to_zarr(store=store)
    # return store.path


@pytest.fixture(scope="session")
def tas_sim(tmp_path_factory):
    ds = tu.tas_sim().to_dataset()
    path = tmp_path_factory.mktemp("data") / "tas_sim.zarr"
    ds.to_zarr(path)
    return path


@pytest.fixture(scope="session")
def hdd_series(tmp_workspace, tas_obs, tas_sim):
    from peach.backend.compute_indicators import (
        ComputeIndicatorsProcessorOBS,
        ComputeIndicatorsProcessorSIM,
    )

    # Compute obs
    p = ComputeIndicatorsProcessorOBS({"name": "Test-HDD-obs"})
    p.INPUT_DATASET_PATH = tas_obs.parent
    p.INPUT_DATASET_PATTERN = tas_obs.name

    data = {
        "name": "HEATING_DEGREE_DAYS",
        "params": {},
        "stations": {"tas": "7028442"},
    }
    mimetype, output = p.execute(data)
    obs = tmp_workspace / output["value"]

    # Compute sim
    p = ComputeIndicatorsProcessorSIM({"name": "Test-HDD-sim"})
    p.INPUT_DATASET_PATH = tas_sim.parent
    p.INPUT_DATASET_PATTERN = tas_sim.name

    mimetype, output = p.execute(data)
    sim = tmp_workspace / output["value"]

    return obs, sim


@pytest.fixture
def synthetic_jp_ds(request):
    ind = request.param if hasattr(request, "param") else "wl_prcond"
    return tu.synthetic_jp_ds(ind)
