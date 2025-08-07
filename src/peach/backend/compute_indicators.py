"""OGC API - Processes - Compute Indicators.

Requires:
 - pygeoapi
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import dask
import s3fs
import xarray as xr
import xclim
from filelock import FileLock, Timeout
from pygeoapi.process.base import BaseProcessor, ProcessorExecuteError

from peach.common import config

# Load water level and IDF indicators into xclim registry
from peach.risk import idf, wl  # noqa: F401

# Dask config set in docker-compose.yml
xclim.set_options(metadata_locales=["fr"])
logger = logging.getLogger("compind")
if (
    config.BUCKET_URL is not None
    and config.BUCKET_CREDENTIALS
    and not config.USE_LOCAL_CACHE
):
    minioFS = s3fs.S3FileSystem(
        use_ssl=config.BUCKET_URL.startswith("https"),
        endpoint_url=config.BUCKET_URL,
        **config.BUCKET_CREDENTIALS,
    )
else:
    minioFS = None


METADATA = {
    "version": 0.1,
    "title": {"en": "Compute indicators", "fr": "Calcul des indicateurs"},
    "description": {
        "en": "Compute climate indicators from input data using xclim",
        "fr": "Calcul des indicateurs climatiques à partir des données d'entrée en utilisant xclim",
    },
    "jobControlOptions": ["async-execute"],
    "keywords": ["climate", "indicators", "xclim"],
    "inputs": {
        "name": {
            "title": {"en": "Indicator", "fr": "Indicateur"},
            "description": {
                "en": "Xclim Indicator ID",
                "fr": "ID d'indicateur de xclim",
            },
            "schema": {"type": "string"},
            "required": True,
            "minOccurs": 1,
            "maxOccurs": 1,
        },
        "params": {
            "title": {"en": "Parameters", "fr": "Paramètres"},
            "description": {
                "en": "Xclim's function arguments",
                "fr": "Arguments de la fonction d'xclim",
            },
            "schema": {"type": "object"},
            "required": True,
            "minOccurs": 1,
            "maxOccurs": 1,
        },
        "stations": {
            "title": {"en": "Station IDs", "fr": "IDs de stations"},
            "description": {
                "en": "Station IDs for each variable.",
                "fr": "IDs de stations pour chaque variable.",
            },
            "schema": {
                "type": "object",
                "properties": {
                    "tas": {"type": "string"},
                    "tasmin": {"type": "string"},
                    "tasmax": {"type": "string"},
                    "pr": {"type": "string"},
                    "idf": {"type": "string"},
                    "wl_pot": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "required": True,
            "minOccurs": 1,
            "maxOccurs": 1,
        },
        "no_cache": {
            "title": {"en": "Deactivate caching.", "fr": "Désactiver la cache."},
            "schema": {"type": "bool"},
            "required": False,
            "minOccurs": 0,
            "maxOccurs": 1,
        },
    },
    "outputs": {
        "links": {
            "title": {"en": "S3 link to the output", "fr": "Lien S3 vers la sortie"},
            "type": "object",
            "description": {
                "en": "S3-compatible link to a Zarr dataset containing the chosen indicator computed on the chosen stations.",
                "fr": "Un lien compatible avec S3 vers un jeu de données Zarr contenent l'indicateur choisi calculé sur les stations choisies.",
            },
        }
    },
    "example": {
        "inputs": {
            "name": "heating_degree_days",
            "params": {"thresh": "17.0 degC", "freq": "YS-JAN"},
            "stations": {"tas": "6105460"},
            "no_cache": True,
        }
    },
}


def _filename_as_real(ds):
    real = "_".join(Path(ds.encoding["source"]).stem.split("_")[1:4])
    return ds.expand_dims(realization=[real])


class ComputeIndicatorsProcessor(BaseProcessor):
    """ComputeIndicatorsProcessor"""

    INPUT_DATASET_PATH = Path("/data")
    INPUT_DATASET_PATTERN = "<please define in subclass>"
    SOURCE = None
    CHECK_MISSING = "any"
    VARIABLES = ["tas", "tasmin", "tasmax", "pr"]

    def __init__(self, processor_def: dict, process_metadata: dict = None):
        """
        Initialize the processor

        :param processor_def: processor definition
        :returns: None
        """
        meta = {"id": f"compute-indicators-{self.SOURCE}", **METADATA}
        meta["title"] = {
            "en": f"Compute indicators from {self.SOURCE}",
            "fr": f"Calcul des indicateurs des {self.SOURCE}",
        }
        if process_metadata:
            meta.update(**process_metadata)

        super().__init__(processor_def, process_metadata=meta)
        self.ds = {}

    @property
    def initialized(self):
        return set(self.ds.keys()) == set(self.VARIABLES)

    @classmethod
    def hash_request(cls, base: str, params: dict, sids: dict):
        """Generate a hash for the given input parameters.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset.
        base : str
            The base indicator to compute.
        params : dict
            The parameters to pass to the base indicator.
        sids : dict
            A dictionary of variable names and station IDs.
        """
        xcver = xclim.__version__.replace(".", "-")
        base_str = base.replace("_", "-")
        params_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()
        ids_str = "-".join([f"{v}{i}" for v, i in sorted(sids.items())])
        return f"{base_str}_{params_hash}_{cls.SOURCE}_{ids_str}_xc{xcver}"

    def open_dataset(self, path: str, pattern: str, data: dict):
        """Open remote or local Zarr datasets."""
        files = [
            get_file(pattern=pattern, path=path, kwds={"var": var})
            for var in self.VARIABLES
        ]

        try:
            for file in files:
                ds = xr.open_dataset(
                    file,
                    # Drop unnecessary stuff
                    drop_variables=[
                        "rotated_pole",
                        "rlon",
                        "rlat",
                        "elev",
                        "frommonth",
                        "fromyear",
                        "prov",
                        "stns_joined",
                        "tomonth",
                        "toyear",
                    ],
                    engine="zarr",
                )
                for var in ds.variables.keys():
                    ds[var].encoding = {}
                logger.debug(f"Data vars: {ds.data_vars.keys()}")
                var = list(set(ds.data_vars.keys()).intersection(self.VARIABLES))[0]
                self.ds[var] = ds[var]
        except FileNotFoundError as err:
            raise ProcessorExecuteError(
                f"Not all input datasets found. Can't find {file}."
            ) from err
        except IndexError:
            raise ProcessorExecuteError(
                f"Input datasets {files} do not contain the required variables: {self.VARIABLES}"
            )

    def execute(self, data):
        """Return indicator time series."""
        logger.info(f"Execution data: {data}")

        if not self.initialized:
            self.open_dataset(self.INPUT_DATASET_PATH, self.INPUT_DATASET_PATTERN, data)
        t0 = time.perf_counter()
        mimetype = "application/json"

        base = data["name"]
        # if "." in base:
        #     mod, indid = base.split(".")
        #     base = f"{mod.lower()}.{indid.upper()}"
        # else:
        #     base = base.upper()
        xc_params = data["params"]

        # Filter out unrelated station information
        stations = {k: v for k, v in data["stations"].items() if k in self.VARIABLES}

        no_cache = data.get("no_cache", config.NO_CACHE_DEFAULT)

        # A unique ID for the request (indicator, params, stations)
        cid = self.hash_request(base, xc_params, stations)
        lock = FileLock(config.WORKSPACE / f"{cid}.lock")
        try:
            # If another process is processing the request, we wait until it's complete
            lock.acquire(timeout=60)
        except Timeout:
            logging.error(
                f"File {cid}.zarr is being written by another process, but it's taking too long. We will forcefully unlock and overwrite."
            )
            lock.release(force=True)
            no_cache = True

        if minioFS is None or config.USE_LOCAL_CACHE:
            outpath = config.WORKSPACE / f"{cid}.zarr"
            exists = outpath.is_dir()
        else:
            outpath = f"{config.BUCKET_NAME}/{cid}.zarr"
            exists = minioFS.exists(outpath)

        if no_cache or not exists:
            out = self.compute_indicator(
                self.ds,
                base,
                xc_params,
                stations,
                self.CHECK_MISSING,
            )
            if minioFS is None or config.USE_LOCAL_CACHE:
                store = outpath
            else:
                store = minioFS.get_mapper(outpath, check=False, create=False)
            out.to_zarr(store, mode="w" if no_cache else "w-")

        t1 = time.perf_counter()
        logger.info(f"{cid} job completed in {t1 - t0} seconds.")
        if minioFS is None or config.USE_LOCAL_CACHE:
            output = {"id": "links", "value": f"{cid}.zarr", "time": t1 - t0}
        else:
            output = {
                "id": "links",
                "value": f"{config.BUCKET_URL}/{config.BUCKET_NAME}/{cid}.zarr",
                "time": t1 - t0,
            }
        lock.release()
        return mimetype, output

    @staticmethod
    def compute_indicator(dss, base, kwds, stations, check_missing):
        """Compute or return cached indicator time series.

        Parameters
        ----------
        dss : dict
            Dictionary of DataArrays.
        base : str
            The base indicator to compute.
        kwds : dict
            The parameters to pass to the base indicator.
        stations : dict
            A dictionary of variable names and station IDs.
        check_missing : str
            Type of checks for missing values, see xclim.
        """
        var, sid = zip(*stations.items())
        inputs = [dss[v].sel(station=s, drop=True) for v, s in zip(var, sid)]

        if len(inputs) > 1:
            inputs = xr.align(*inputs)
        inputs = dict(zip(var, dask.compute(*inputs)))
        reg = xclim.core.indicator.registry[base]
        ind = reg.get_instance()
        indexer = kwds.pop("indexer", {})
        with xclim.set_options(check_missing=check_missing):
            out = ind(**inputs, **kwds, **indexer)
        if out.attrs["units"] == "K":
            logger.info(f"Converting indicator {out.name} from K to °C")
            out = xclim.core.units.convert_units_to(out, "°C")
        out.attrs["id"] = base
        out.attrs["params"] = kwds
        out.attrs["stations"] = stations
        if "long_name_fr" not in out.attrs.keys():
            raise ValueError(f"Translation not found on output of {out.name}")

        if "realization" in out.dims:
            out = out.chunk(realization=100)
        return out

    def __repr__(self):
        """Return string representation."""
        return f"<Processor> {self.name}"


class ComputeIndicatorsProcessorOBS(ComputeIndicatorsProcessor):
    INPUT_DATASET_PATTERN = os.environ.get(
        "OBS_PATTERN", "AHCCD_{var}_stations-ping.zarr"
    )
    SOURCE = "obs"
    CHECK_MISSING = "pct"


class ComputeIndicatorsProcessorSIM(ComputeIndicatorsProcessor):
    INPUT_DATASET_PATTERN = os.environ.get("SIM_PATTERN", "NEW_SIMS_{var}_day.zarr")
    SOURCE = "sim"
    CHECK_MISSING = "skip"


# Water level backend computation
# The xclim indicator is essentially a lambda function, so it returns whatever is loaded in `open_dataset`.


class ComputeWaterLevelProcessorOBS(ComputeIndicatorsProcessor):
    INPUT_DATASET_PATTERN = os.environ.get("OBS_WL_PATTERN", "*_wl_pot.nc")
    SOURCE = "wl-obs"
    CHECK_MISSING = "skip"
    VARIABLES = ["wl_pot"]

    def open_dataset(self, path: str, pattern: str, data: dict):
        """Open remote or local Zarr dataset."""
        var = self.VARIABLES[0]  # Variable name in station dictionary
        station_id = data["stations"][var]

        fs = get_file(
            pattern=pattern, path=path, kwds={"var": var, "station_id": station_id}
        )
        ds = xr.open_dataset(fs)

        logger.debug(f"Data vars: {ds.data_vars.keys()}")

        # Add station dimension for compatibility with compute_indicator method
        da = ds[var].expand_dims(station=[station_id])

        self.ds[var] = da


class ComputeWaterLevelProcessorSIM(ComputeWaterLevelProcessorOBS):
    INPUT_DATASET_PATTERN = os.environ.get("SIM_WL_PATTERN", "*_sl.nc")
    SOURCE = "wl-sim"
    CHECK_MISSING = "skip"

    def open_dataset(self, path: str, pattern: str, data: dict):
        """Open remote or local Zarr dataset."""
        var = self.VARIABLES[0]  # Variable name in station dictionary
        station_id = data["stations"][var]

        fs = get_file(
            pattern=pattern, path=path, kwds={"var": "sl", "station_id": station_id}
        )
        ds = xr.open_dataset(fs)

        logger.debug(f"Data vars: {ds.data_vars.keys()}")

        # Add station dimension for compatibility with compute_indicator method
        da = ds["sl_delta"]
        da = da.expand_dims(station=[station_id])

        self.ds[var] = da


class ComputeIDFProcessorOBS(ComputeIndicatorsProcessor):
    INPUT_DATASET_PATTERN = os.environ.get("OBS_IDF_PATTERN", "IDF3.30.zarr")
    SOURCE = "idf-obs"
    CHECK_MISSING = "skip"
    VARIABLES = ["idf"]


class ComputeIDFProcessorSIM(ComputeIndicatorsProcessorSIM):
    SOURCE = "idf-sim"
    CHECK_MISSING = "skip"
    VARIABLES = ["tas"]

    @staticmethod
    def compute_indicator(dss, base, kwds, stations, check_missing):
        """Compute or return cached indicator time series.

        Parameters
        ----------
        dss : dict
            Dictionary of DataArrays.
        station_id : dict
            A dictionary of variable names and station IDs.
        base : str
            The base indicator to compute.
        parameters : dict
            The parameters to pass to the base indicator.
        """
        # Override the IDF base indicator (a lambda function) to instead compute the mean annual temperature for the
        # CC scaling.
        base = "TG_MEAN"
        kwds.pop("duration")
        logger.debug("Computing mean annual temperature for IDF scaling.")

        out = ComputeIndicatorsProcessorSIM.compute_indicator(
            dss, base, kwds, stations, check_missing
        ).rename("idf")
        # TODO Override more metadata ??
        # out.name = "idf"
        return out


def get_file(pattern: str, path: str, kwds: dict):
    """Return a path to a local file or a link to an S3 file."""
    if pattern.startswith("s3://"):
        # minio / s3fs
        url = urlparse(pattern[5:])

        s3r = s3fs.S3FileSystem(
            anon=True, use_ssl=False, endpoint_url=f"{url.scheme}://{url.netloc}"
        )
        link = url.path.format(**kwds)
        if link.endswith(".zarr"):
            return s3fs.S3Map(root=link, s3=s3r, check=False)
        elif link.endswith(".nc"):
            return s3r.open(link)

    else:
        return Path(path) / pattern.format(**kwds)
