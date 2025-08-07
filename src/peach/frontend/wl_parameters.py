"""
# Parameter classes for the panel application, specific to water level indicators.

- IndicatorObsWL: Water level parameterized class, customized for Peaks Over Threshold (POT) values.
- IndicatorSimWL: Same but with sea-level simulations used to shift the extreme value distribition.
- IndicatorRefWL: Same but returns the reference period.
"""

import numpy as np
import param
import xarray as xr
import xclim as xc
from lmoments3.distr import gpa

from peach.common import config
from peach.frontend.cbcl_utils import wl_norm
from peach.frontend.parameters import (
    TYPE_MAP,
    VARIABLE_MAPPING,
    IndicatorDA,
    scen_weights,
)
from peach.risk import bootstrap
from peach.risk.xmixture import XMixtureDistribution

"""
# Design considerations

We need different types of "data" here.

## Observations

What we have are POT values, detrended and normalized to a reference period (AR6 reference period, 1995-2014). We'll
call this the computed data and store it in the `_data` attribute.

What we also need are:
  D1. Observation data for plotting - including trends. This is affected by the choice of period:
    `POT(t) + sl_mm_yr * (t - AR6_ref_midpoint) / 1000`
  D2. Observation data for fitting distributions (no trend, but need for renormalization if period is different from AR6
  ref):
    `POT(t) + sl_mm_yr * (period_midpoint - AR6_ref_midpoint) / 1000`

What we propose is to provide D2 using the `sample` property, and D1 using the `data` attribute.

## Simulations

What we have are the SLR CDF values relative to [1995, 2014] for different scenarios at 10-year intervals. We can
generate random sample from this CDF using the `bootstrap.from_quantile` function.

What we need are:
    1. Data for plotting - including trends from the SLR projections for each SSP (10p, 50p, 90p). We
    need to add `stn_thresh` to those values so the plots are comparable to observations.
    2. Data for fitting distributions - Because we're using a delta, we don't really account for natural variability
    in the fit of the distribution. To make up for this, we account for parametric uncertainty by doing a bootstrap
    resampling of the observations and fitting the distribution to each sample. We then modify the loc parameter by
    the SLR interpolated at the mid point of the period and renormalized relative to the observation period. The
    mixture distribution then weight the different SSPs and bootstrap samples to compute a single number for
    different statistics.
        `SLR(mid_period) - sl_mm_yr * (obs.period_midpoint - AR6_ref_midpoint)`
"""


class IndicatorObsWL(IndicatorDA):
    """Water level parameterized class, customized for Peaks Over Threshold (POT) values.

    Notes
    -----
    POT values are computed from a detrended time series of water levels, and adjusted to a reference period.
    To get water levels for another period, we need to adjust the POT values to the new period.

    We may bootstrap observations as well to account for parametric uncertainty in the future. I think all the
    mechanics are in place to do this fairly easily.
    """

    dist = param.Selector(default="genpareto", objects=["genpareto"])
    data = param.Parameter(
        doc="DataArray time series of detrended, normalized peak-over-threshold values.",
        constant=True,
    )
    long_name = param.Parameter("Water level peaks over threshold")

    stn_thresh = param.Number(doc="Station threshold used to compute POT values.")
    peaks_per_yr = param.Number(doc="Number of peaks per year.")
    sl_mm_yr = param.Number(doc="Sea level rise rate (mm/y).")
    ref_period = param.Tuple(doc="Reference period for normalization.")

    @param.depends("data", watch=True, on_init=True)
    def _update_attrs(self):
        # Get some of the parameters from the data attributes.
        for key in ["stn_thresh", "peaks_per_yr", "sl_mm_yr"]:
            setattr(self, key, self.data.attrs[key])

        setattr(self, "ref_period", tuple(self.data.attrs["ref_period"]))

    @param.depends("data", "ref_period", "sl_mm_yr", watch=True, on_init=True)
    def _update_ts(self):
        """Add trend to data."""
        self.ts = add_trend(
            self.data,
            slope=self.sl_mm_yr / 1000,
            midpoint=np.mean(self.ref_period),
        )

    @property
    def ts_caption(self):
        stations = []
        for vv, sid in self.data.stations.items():
            row = config.get_station_meta(sid, VARIABLE_MAPPING.get(vv, vv))
            stations.append(f"{row.station_name} ({sid})")

        thresh = self.data.attrs.get("stn_thresh", self.stn_thresh)

        if len(stations) == 1:
            st = "station " + stations[0]
        else:
            st = "stations " + ", ".join(stations)

        if self.locale == "fr":
            if len(stations) == 1:
                st = "à la " + st
            else:
                st = "aux " + st

            return f"Série observée de `{self.long_name}` {st}, avec un seuil de {thresh} m."
        else:
            return f"Observed time series for `{self.long_name}` at {st}, with a threshold of {thresh} m."

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

    def _sample(self, period: tuple) -> xr.DataArray:
        """Return the full detrended time series, renormalized over the desired period."""
        with xr.set_options(keep_attrs=True):
            return self.data + wl_norm(self.sl_mm_yr, period)

    def _dist_method(self, name, arg: xr.DataArray | float):
        """Adjust argument or results for the fact that the pareto parameters describe events that may
        occur more than once per year on average. If there are 2 event per year, then the return period is halved.
        """
        func = lambda x: xc.indices.stats.dist_method(
            function=name, fit_params=self.dparams, arg=x
        )
        return scale_pareto(func, name, arg, self.peaks_per_yr)

    def pdf(self, x):
        """Return the probability density function."""
        pdf = lambda x: xc.indices.stats.dist_method(
            function="pdf", fit_params=self.dparams, arg=x
        )
        sf = lambda x: xc.indices.stats.dist_method(
            function="sf", fit_params=self.dparams, arg=x
        )
        n = self.peaks_per_yr
        out = n * (1 - sf(x)) ** (n - 1) * pdf(x)
        out.name = "pdf"
        return out

    def fit(self, dist: str, period: tuple, size=None, iteration=1) -> xr.DataArray:
        """Fit the distribution to the data (normalized for period) over the full period."""
        # TODO v2: Account for parametric uncertainty in the observations.

        if dist != "genpareto":
            raise ValueError(f"Only genpareto is supported, not {dist}.")

        sample = self._sample(period)

        if iteration > 1:
            if size is None:
                size = len(sample)

            sample = bootstrap.resample(
                sample, size=size, iteration=iteration, replace=True
            )

        # Fit the parameters
        # TODO: floc or not ?
        dparams = xc.indices.stats.fit(
            sample,
            dist=gpa,
            dim="time",
            method="PWM",
        )

        return dparams


class IndicatorSimWL(IndicatorDA):
    """Water level parameterized class, custom made for POT values.

    Bootstrap used to combine sampling uncertainties with climate change uncertainties.

    The important method here is `sf` (1-cdf).
    """

    dist = param.Selector(default="genpareto", objects=["genpareto"], allow_refs=True)
    obs = param.ClassSelector(class_=IndicatorObsWL, doc="Observed indicator.")
    data = param.Parameter(doc="AR6 sea level rise rate DataArray.")
    model_weights = param.Parameter(
        doc="Weights for the models. Unused here.", allow_refs=True
    )
    weights = param.Parameter(doc="Weights for the scenarios")
    num_samples = param.Number(150, doc="Number of samples to draw for the bootstrap.")
    sample_size = param.Number(30, doc="Size of the bootstrap samples.")

    def __init__(self, **kwargs):
        IndicatorDA.__init__(self, **kwargs)

    def _sample(self, period: tuple) -> xr.DataArray:
        """Return the SLR delta relative to the obs period."""
        year = str(int(np.mean(period)))
        slr = self.data.interp(time=year)

        with xr.set_options(keep_attrs=True):
            return slr - wl_norm(self.obs.sl_mm_yr, self.obs.period)

    @property
    def sample(self):
        """Return sea level rise at period mid-point across all scenarios and percentiles, scaled by the observation
        threshold.
        """
        with xr.set_options(keep_attrs=True):
            return self._sample(self.period) + self.obs.stn_thresh

    def fit(self, dist: str, period: tuple) -> xr.DataArray:
        """Fit the distribution to the data, creating a bootstrap sample combining parametric uncertainty for the
        observation data, and scenario uncertainty for the future data.
        """
        # Bootstrapped distribution parameters for the observations, renormalized.
        dparams = self.obs.fit(
            dist, self.obs.period, size=self.sample_size, iteration=self.num_samples
        )
        dp = dparams.expand_dims(experiment_id=self.data.experiment_id)

        # Bootstrapped future sea level rise - mid-year
        slr = self._sample(period)
        if isinstance(slr, float):
            slr_sample = slr
        else:
            slr_sample = bootstrap.from_quantile(slr, iteration=self.num_samples)

        # Adjust loc parameter from obs to include future sea level rise
        loc = dp.sel(dparams="loc") + slr_sample
        dp_copy = dp.copy()
        dp_copy.loc[dict(dparams="loc")] = loc
        return dp_copy

    def pdf(self, x):
        """Return the probability density function."""
        mix = XMixtureDistribution(params=self.dparams, weights=self.weights)

        n = self.obs.peaks_per_yr
        out = n * (1 - mix.sf(x)) ** (n - 1) * mix.pdf(x)
        out.name = "pdf"
        return out

    def _dist_method(self, name, arg: xr.DataArray | float):
        """Compute weighted statistics."""
        mix = XMixtureDistribution(params=self.dparams, weights=self.weights)
        func = getattr(mix, name)
        return scale_pareto(func, name, arg, self.obs.peaks_per_yr)

    @param.depends("period", watch=True, on_init=True)
    def _update_weights(self):
        """Compute the scenario weights over the period."""
        if max(self.period) > 2100:
            raise ValueError("Period must end before 2100.")

        # Scenario weights
        scen_w = scen_weights.sel(time=self._slice(self.period)).mean(dim="time")

        # Outside of the SSP period, assume all scenarios equally likely.
        if scen_w.isnull().all():
            scen_w = scen_w.fillna(0.25)

        # Sample weights
        sample_w = xr.DataArray(
            np.ones(self.num_samples) / self.num_samples, dims=("sample")
        )

        # Combined weights
        self.weights = scen_w * sample_w

    def experiment_percentiles(self, per) -> xr.Dataset:
        """Return the percentiles computed for each year and experiment.

        Useful for visualizing the distribution of the ensemble.

        Parameters
        ----------
        per : list of int
          The percentiles to compute [0, 100].
        """
        per = np.asarray(per) / 100
        out = self.data.interp(quantile=per) + self.obs.stn_thresh

        # Harmonize outputs with IndicatorSimDA.experiment_percentiles.
        out = out.to_dataset(dim="quantile")
        for p, perc in out.data_vars.items():
            perc.attrs["description"] = (
                perc.attrs.get("description", "") + f" {p}th percentile of ensemble."
            )
            out[p] = perc
            out = out.rename(name_dict={p: f"{self.data.name}_p{int(p*100):02d}"})

        return out

    @param.depends("obs", watch=True, on_init=True)
    def _update_title(self):
        """Set default long_name from indicator."""
        if self.long_name == "" and self.obs:
            self.long_name = self.obs.long_name

    @property
    def ts_caption(self):
        stations = []
        for vv, sid in self.data.stations.items():
            row = config.get_station_meta(sid, VARIABLE_MAPPING.get(vv, vv))
            stations.append(f"{row.station_name} ({sid})")

        thresh = self.obs.stn_thresh

        if len(stations) == 1:
            st = "station " + stations[0]
        else:
            st = "stations " + ", ".join(stations)

        if self.locale == "fr":
            if len(stations) == 1:
                st = "à la " + st
            else:
                st = "aux " + st

            return (
                f"Série projetée de `{self.long_name}` {st}, avec un seuil de {thresh} m. Cliquer sur les items de la légende "
                f"permet de contrôler leur visibilité."
            )
        else:
            return f"Projected time series for `{self.long_name}` at {st}, with a threshold of {thresh} m. Clicking on legend items allows to control their visibility."

    @property
    def hist_caption(self):
        period = "–".join(map(str, self.period))
        dist = self.dist
        if self.locale == "fr":
            return (
                f"Densité de probabilité de la distribution {dist}, superposée à l'histogramme de la série "
                f"projetée au cours de la période ({period})."
            )
        return (
            f"Probability density function of the {dist} distribution, overlaid on the histogram of the projected "
            f"series during the period ({period})."
        )


class IndicatorRefWL(IndicatorSimWL):
    level = param.Number(
        0.1, doc="Significance level for the KS test. Unused here.", allow_refs=True
    )

    @property
    def period(self):
        return self.obs.period

    def _sample(self, period: tuple) -> xr.DataArray:
        """No  climate change signal during the Ref period."""
        return -wl_norm(self.obs.sl_mm_yr, self.obs.period)

    @param.depends("data", watch=True, on_init=True)
    def _update_ks(self):
        """Return an array of ones to include all models in computation.
        TODO: Ideally we'd run a KS test against temperature observations over the reference period.
        """
        self._ks = True


def add_trend(da: xr.DataArray, slope: float, midpoint: float) -> xr.DataArray:
    """Add a linear trend to the data array.

    Parameters
    ----------
    da : xr.DataArray
      The data array to add the trend to.
    slope : float
      The slope of the trend.
    midpoint : float
      The midpoint of the trend, in years.

    Returns
    -------
    xr.DataArray
      The data array with the trend added.
    """
    with xr.set_options(keep_attrs=True):
        out = da + slope * (da.time.dt.year - midpoint)
        out.name = da.name
        return out


def scale_pareto(func, name, arg, peaks_per_yr):
    """Scale statistics of the pareto distribution to correct for the fact that the parameters are fitted to
    a sample with not exactly one event per year.

    Parameters
    ----------
    func : Callable
        Function taking argument `arg` and returning a statistic based on a distribution fitted to a sample with,
        on average, one event per year.
    name : str
        Name of statistical method.
    arg : float or xr.DataArray
        Argument of the method.
    peaks_per_yr : float
        Number of events per year going into the fit of the parameters.

    Notes
    -----
    Let's denote sf' as the survival function for the number of peaks per year, and sf as the survival function for
    the normalized sample with, on average, one peak per year.

    sf'(x) = peaks_per_yr * sf(x)
    isf'(x) = isf(x / peaks_per_yr)
    """
    if name not in ["sf", "isf"]:
        raise ValueError(f"Method {name} not supported.")

    if name in ["isf"]:
        arg = arg / peaks_per_yr

    out = func(arg)

    if name in ["sf"]:
        out = out * peaks_per_yr

    try:
        out.name = name
    except AttributeError:
        pass
    return out


TYPE_MAP["obs"]["wl_pot"] = IndicatorObsWL
TYPE_MAP["sim"]["wl_pot"] = (IndicatorRefWL, IndicatorSimWL)
TYPE_MAP["sim"]["sl_delta"] = (IndicatorRefWL, IndicatorSimWL)
