"""
# Functions to interact with the config.
"""

import json
import os
from pathlib import Path

import pandas as pd
import yaml
import warnings

SERVICE = os.environ.get("SERVICE")
# LOCAL DIR is the root of the python source (src/portail_ing/)
LOCAL_DIR = Path(__file__).parent.parent.resolve()
CONFIG_DIR = Path(
    os.environ.get("FRONTEND_CONFIG_DIR", LOCAL_DIR / "frontend" / "config")
)
DATA_DIR = Path(os.environ.get("FRONTEND_DATA_DIR", LOCAL_DIR / "frontend" / "data"))


# Default workspace is folder besides the root src/
WORKSPACE = Path(os.environ.get("WORKSPACE", LOCAL_DIR.parent.parent / "workspace"))
BUCKET_URL = os.environ.get("BUCKET_URL")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
BUCKET_CREDENTIALS_FILE = os.environ.get("BUCKET_CREDENTIALS_FILE")
USE_LOCAL_CACHE = str(os.environ.get("USE_LOCAL_CACHE", "1")).lower() in [
    "true",
    "1",
    "t",
    "y",
    "yes",
]
print(
    f"config.py - Using local cache: {USE_LOCAL_CACHE}, with file {BUCKET_CREDENTIALS_FILE}"
)

BUCKET_CREDENTIALS = {}
if BUCKET_CREDENTIALS_FILE and not USE_LOCAL_CACHE:
    with open(BUCKET_CREDENTIALS_FILE) as f:
        cre = json.load(f)
    if (cre.get("accessKey") is None) or (cre.get("secretKey") is None):
        USE_LOCAL_CACHE = True
        warnings.warn(f"Using Local Cache despite BUCKET_CREDENTIALS_FILE={BUCKET_CREDENTIALS_FILE} and not USE_LOCAL_CACHE={USE_LOCAL_CACHE}")
    else:
        BUCKET_CREDENTIALS["key"] = cre.get("accessKey")
        BUCKET_CREDENTIALS["secret"] = cre.get("secretKey")


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost/")
if not BACKEND_URL.endswith("/"):
    BACKEND_URL = BACKEND_URL + "/"

MAX_RETRIES = os.environ.get("MAX_RETRIES", 2)
_stations = {}

NO_CACHE_DEFAULT = False
MAX_INDICATORS = os.environ.get("MAX_INDICATORS", 10)

MIN_OBS_DATA = 30


def read_stations():
    return pd.read_csv(DATA_DIR / "stations_variables.csv").round(3)


def get_station_meta(station_id, variable):
    if SERVICE == "frontend":
        import panel as pn

        df_sta = pn.state.cache.get("df_sta", None)
        if df_sta is None:
            df_sta = read_stations()
            pn.state.cache["df_sta"] = df_sta
    else:
        if "df" not in _stations:
            _stations["df"] = read_stations()
        df_sta = _stations["df"]

    return df_sta.set_index(["station", "variable"]).loc[(station_id, variable)]


def read_indicator_config():
    with open(CONFIG_DIR / "indicators.yml") as f:
        ic = yaml.safe_load(f)
    return ic


def read_application_config():
    with open(CONFIG_DIR / "application.yml") as f:
        ac = yaml.safe_load(f)
    return ac
