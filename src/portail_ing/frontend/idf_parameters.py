import json
from pathlib import Path

import numpy as np
import param
import xarray as xr
import xclim as xc

from portail_ing.common import config
from portail_ing.common.logger import get_logger
from portail_ing.frontend.parameters import (
    TYPE_MAP,
    VARIABLE_MAPPING,
    IndicatorDA,
    IndicatorRefDA,
    IndicatorSimDA,
)
from portail_ing.risk.priors import members

logger = get_logger(__name__)

# Source: Procédure proposée pour estimer les intensités des maxima annuels de
# précipitations (MAP) en climat futur, Mailhot et al., sent 2024-08-16
# TODO: duration_coeff have changed with the latest version of the document

# Coefficients for the IDF scaling (%/°C)
regional_coeff = {
    1: 4.2,
    2: 2.1,
    3: 1.9,
    4: 2.1,
    5: 2.6,
    6: 3.4,
    7: 3.8,
    8: 4.8,
    9: 3.6,
    10: 2.6,
    11: 3.1,
    12: 2.8,
    13: 3.2,
    14: 3.9,
    15: 3.3,
}

duration_coeff = {"1h": 1.51, "2h": 1.44, "6h": 1.24, "12h": 1.11, "24h": 1.0}

# Load regions file
regions = json.load(open(Path(__file__).parent / "data" / "idf_regions.json"))


class IndicatorObsIDF(IndicatorDA):
    dist = param.Selector(default="gumbel_r", objects=["gumbel_r"])
    duration = param.String(doc="Rainfall duration")
    station_id = param.String(doc="Station ID")

    @param.depends("data", watch=True, on_init=True)
    def _update_attrs(self):
        self.duration = self.data.duration.item()
        self.station_id = self.data.attrs["stations"]["idf"]

    @param.depends("data", watch=True, on_init=True)
    def _update_ts(self):
        """Set `ts` to mm of rain."""
        with xr.set_options(keep_attrs=True):
            self.ts = self.data * xc.units.units(self.duration).to("h").magnitude * 1000
            self.ts.attrs["units"] = "mm"

    @property
    def ts_caption(self):
        stations = []
        for vv, sid in self.data.stations.items():
            row = config.get_station_meta(sid, VARIABLE_MAPPING.get(vv, vv))
            stations.append(f"{row.station_name} ({sid})")

        if len(stations) == 1:
            st = "station " + stations[0]
        else:
            st = "stations " + ", ".join(stations)

        if self.locale == "fr":
            if len(stations) == 1:
                st = "à la " + st
            else:
                st = "aux " + st

            return f"Série observée de `{self.long_name}` {st}. {self.description}"
        else:
            return f"Observed time series for `{self.long_name}` at {st}. {self.description}"

    @property
    def hist_caption(self):
        period = "–".join(map(str, self.period))
        dist = self.dist
        if self.locale == "fr":
            return (
                f"Densité de probabilité de la distribution {dist}, superposée à l'histogramme de la série "
                f"observée au cours de la période ({period})."
            )
        return (
            f"Probability density function of the {dist} distribution, overlaid on the histogram of the observed "
            f"series during the period ({period})."
        )


class IndicatorSimIDF(IndicatorSimDA):
    dist = param.Selector(default="gumbel_r", objects=["gumbel_r"])

    obs = param.ClassSelector(class_=IndicatorObsIDF, doc="Observed indicator.")

    data = param.Parameter(
        doc="Mean annual temperature time series.",
    )
    title = param.Parameter("Extreme rainfall")

    def __init__(self, **kwargs):
        self.data = (
            kwargs["data"]
            .set_index(realization=("variant_label", "source_id", "experiment_id"))
            .unstack("realization")
        )
        super().__init__(**kwargs)
        self._value = {
            "long_name": self.obs._value["long_name"],
            "description": self.obs._value["description"],
        }

    @param.depends("data", watch=True, on_init=True)
    def _update_attrs(self):
        self.regional_coeff = regions[self.obs.station_id]
        self.duration_coeff = duration_coeff[self.obs.duration]
        self.scaling_factor = 1 + self.regional_coeff * self.duration_coeff / 100

    @param.depends("analysis.fut_period", "analysis.ref_period", "data", watch=True, on_init=True)
    def _update_ts(self):
        """Store the scaled precipitation values relative to the observation period in `data`."""
        # Compute temperature difference with respect to obs period
        with xr.set_options(keep_attrs=True):
            tg = self.data.rolling(time=30, center=True).mean()
            delta_tas = tg - self.data.sel(time=self._slice(self.obs.period)).mean(
                "time"
            )
        delta_tas.attrs["units_metadata"] = "temperature: difference"

        pr = (
            self.obs.data.sel(time=self.obs._slice(self.obs.period))
            .dropna("time")
            .rename(time="t")
        )

        out = xc.indices.clausius_clapeyron_scaled_precipitation(
            delta_tas, pr_baseline=pr, cc_scale_factor=self.scaling_factor
        )

        with xr.set_options(keep_attrs=True):
            out = out * xc.units.units(self.obs.duration).to("h").magnitude * 1000
            out.attrs["units"] = "mm"

        self.ts = out

    def delta(self, period: tuple) -> xr.DataArray:
        """Return the temperature difference between a period and the observation period.

        Parameters
        ----------
        period : tuple
          The period to compute the difference for.
        """
        with xr.set_options(keep_attrs=True):
            delta = self.data.sel(time=self._slice(period)).mean(
                dim="time"
            ) - self.data.sel(time=self._slice(self.obs.period)).mean(dim="time")
        return delta.assign_attrs(units_metadata="temperature: difference")

    def _sample(self, period: tuple) -> xr.DataArray:
        """Return the extreme rainfall values scaled for the given period."""
        # Compute the temperature difference between the period and the observation period
        delta_tas = self.delta(period)

        # Compute the scaled precipitation
        # TODO: Use all the data or just the sample from the reference period?
        pr = self.obs.data.sel(time=self.obs._slice(self.obs.period)).dropna("time")
        out = xc.indices.clausius_clapeyron_scaled_precipitation(
            delta_tas, pr_baseline=pr, cc_scale_factor=self.scaling_factor
        )

        # Update the time coordinate
        ny = float(np.mean(period) - np.mean(self.obs.period))
        out["time"] = out.time + np.timedelta64(int(ny * 365.25), "D")

        with xr.set_options(keep_attrs=True):
            out = out * xc.units.units(self.obs.duration).to("h").magnitude * 1000
            out.attrs["units"] = "mm"
            out.name = self.obs.data.name

        return out

    def experiment_percentiles(self, per) -> xr.Dataset:
        """Return the percentiles computed for each year and experiment.

        Useful for visualizing the distribution of the ensemble.

        Parameters
        ----------
        per : list of int
          The percentiles to compute [0, 100].
        """
        if self.model_weights is None:
            raise ValueError("Please set the model weights.")

        da = self.ts.stack(realization=["source_id", "variant_label", "t"])

        # Apply weights on the ensemble percentile calculations
        # Here we need to apply weights related to the number of members, as well as model weights
        w = self.model_weights * members(self.data)
        w = w.expand_dims(variant_label=self.ts.variant_label, t=self.ts.t).stack(
            realization=["source_id", "variant_label", "t"]
        )
        w = w.fillna(0)
        # Mismatch problem here between realization dimensions (da includes t) and weights (w does not)
        # TODO: I think xclim is buggy when len(per) == 1. Check. If so, we need to fix it.
        return xc.ensembles.ensemble_percentiles(da, per, split=True, weights=w)


class IndicatorRefIDF(IndicatorSimIDF, IndicatorRefDA):

    level = param.Number(
        0.1, doc="Significance level for the KS test. Unused here.", allow_refs=True
    )

    @param.depends("data", watch=True, on_init=True)
    def _update_ks(self):
        """Return an array of ones to include all models in computation.
        TODO: Ideally we'd run a KS test against temperature observations over the reference period.
        """
        self._ks = ~self.data.source_id.isnull()

        # The subclassing seems to break the dependency, so we need to call it manually
        self._update_model_weights()


TYPE_MAP["obs"]["idf"] = IndicatorObsIDF
TYPE_MAP["sim"]["idf"] = (IndicatorRefIDF, IndicatorSimIDF)
