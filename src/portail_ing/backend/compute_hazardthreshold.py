"""OGC API - Processes - Compute Hazard Thresholds"""

import hashlib
import logging
import time

import xclim
from pygeoapi.process.base import BaseProcessor

from portail_ing.common import config

# TODO move params to common
from portail_ing.frontend import parameters as params

# Load water level and IDF indicators into xclim registry

# Dask config set in docker-compose.yml
logger = logging.getLogger("comphazard")


METADATA = {
    "version": 0.1,
    "title": {"en": "Compute hazard thresholds", "fr": "Calcul des seuils d'aléas"},
    "description": {
        "en": "Compute hazard thresholds",
        "fr": "Calcul des seuils d'aléas",
    },
    "jobControlOptions": ["async-execute"],
    "keywords": ["climate", "indicators", "xclim"],
    "inputs": {
        "indicators": {
            "title": {"en": "Indicators", "fr": "Indicateurs"},
            "description": {
                "en": "List of indicator definitions (objects with entries 'name' and 'params', the same as in the indicator computation).",
                "fr": "Liste de définitions d'indicateurs (objets avec des clés 'name' et 'params', les mêmes que pour le calcul des indicateurs).",
            },
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "params": {"type": "object"},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "uniqueItems": True,
            },
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
        "analysis": {
            "title": {"en": "Analysis Parameters", "fr": "Paramètres de l'analyse"},
            "description": {
                "en": "Can contain: ref_period (int tuple), fut_period (int tuple), dist (list of strings, same length as 'indicators'), metric (pareil que dist).",
                "fr": "Peut contenir: ref_period (int tuple), fut_period (int tuple), dist (liste de strings de même longueur que 'indicators'), metric (pareil que dist).",
            },
            "schema": {
                "type": "object",
                "properties": {
                    "ref_period": {"type": "array", "items": {"type": "integer"}},
                    "fut_period": {"type": "array", "items": {"type": "integer"}},
                    "dist": {"type": "array", "items": {"type": "string"}},
                    "metric": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            "required": True,
            "minOccurs": 1,
            "maxOccurs": 1,
        },
        "hazards": {
            "title": {"en": "Hazards list", "fr": "Liste d'aléas"},
            "description": {
                "en": (
                    "List of hazards, each element in a list with objects with an optional 'description' "
                    "(defaults to 'Aléa #N') and one of X (threshold) or T (return period) (not both)."
                ),
                "fr": (
                    "Liste d'aléas, chaque élément est une liste d'objets avec une 'description' optionnelle "
                    "('Aléa #N') et une entrée entre X (seuil) ou T (période de retour) (une seule des deux)."
                ),
            },
            "schema": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "X": {"type": "number"},
                            "T": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "required": True,
            "minOccurs": 1,
            "maxOccurs": 1,
        },
    },
    "outputs": {
        "output": {
            "title": "Output",
            "type": "object",
        }
    },
}
ind_config = config.read_indicator_config()


class ComputeHazardThreshold(BaseProcessor):
    """ComputeHazardThreshold"""

    def __init__(self, processor_def: dict, process_metadata: dict = None):
        """
        Initialize the processor

        :param processor_def: processor definition
        :returns: None
        """
        meta = {"id": "compute-hazardthreshold", **METADATA}
        meta["title"] = {
            "en": "Compute hazard thresholds",
            "fr": "Calcul des seuils de danger",
        }
        if process_metadata:
            meta.update(**process_metadata)

        super().__init__(processor_def, process_metadata=meta)

    @classmethod
    def hash_request(cls, indicators: list, sids: dict, hazards: dict):
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
        inds_str = [
            f"{ind['name'].replace('_', '-')}_{hashlib.md5(str(sorted(ind.get('params', {}).items())).encode()).hexdigest()}"
            for ind in indicators
        ]
        ids_str = "-".join([f"{v}{i}" for v, i in sorted(sids.items())])
        hazard_str = "-".join([f"{v}{i}" for v, i in sorted(hazards.items())])
        return f"{inds_str}_{ids_str}_{hazard_str}_xc{xcver}"

    def execute(self, data):
        """Return indicator time series."""
        logger.info(f"Execution data: {data}")

        t0 = time.perf_counter()
        mimetype = "application/json"

        # Indicator container, configure it with only the needed indicator to save time
        indicators = params.IndicatorList(
            backend=config.BACKEND_URL,
            station_id=data["stations"],
            config={ind["name"]: ind_config[ind["name"]] for ind in data["indicators"]},
        )
        uuids = indicators.select_from_list(data["indicators"])

        # Compute
        indicators.post_all_requests()
        while indicators.monitor_jobs() is False:
            # TODO Timeout ?
            time.sleep(2)

        # Analysis
        an_kwargs = {
            k: tuple(v)
            for k, v in data["analysis"].items()
            if k in ["ref_period", "fut_period"]
        }
        analysis = params.Analysis(indicators=indicators, **an_kwargs)
        analysis.update_params(
            **{
                k: dict(zip(uuids, data["analysis"][k]))
                for k in ["metric", "dist"]
                if k in data["analysis"]
            }
        )

        # Hazards
        matrix = params.HazardMatrix(analysis=analysis)
        hazard_dict = dict(zip(uuids, data["hazards"]))
        matrix.from_dict(hazard_dict)

        t1 = time.perf_counter()
        logger.info(f"Job completed in {t1 - t0} seconds.")
        anaout = analysis.to_dict("short")
        matout = matrix.to_dict()
        output = {
            "value": {"analysis": anaout, "hazards": [matout[uuid] for uuid in uuids]}
        }
        logger.info(f"Job done : {output}")
        return mimetype, output

    def __repr__(self):
        """Return string representation."""
        return f"<Processor> {self.name}"
