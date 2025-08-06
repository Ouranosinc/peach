"""
Joint probability parameters classes for the portail_ing frontend.

Note that the 'dist' naming was kept from parent classes, but here it refers to copula families,
not scipy univariate distributions.
"""

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import param
import xarray as xr
import xclim as xc
from lmoments3.distr import gpa
from xclim.core.formatting import update_history

from portail_ing.frontend.cbcl_utils import define_q, matching_events, wl_norm
from portail_ing.frontend.copula import (
    check_param,
    copulae_copula,
    optim_options,
    ot_aic,
    ot_bic,
    ot_copula,
    pobs,
)
from portail_ing.frontend.parameters import IndicatorDA, IndicatorObsDA, scen_weights
from portail_ing.frontend.wl_parameters import (
    IndicatorObsWL,
    IndicatorSimWL,
    add_trend,
    scale_pareto,
)
from portail_ing.risk import bootstrap
from portail_ing.risk.priors import model_weights_from_sherwood
from portail_ing.risk.xmixture import XMixtureDistribution

copulas = ["gaussian", "student", "clayton", "frank", "gumbel", "indep"]
scipy_dists = ["norm", "t", "gamma", "genextreme", "lognorm", "uniform"]


class IndicatorObsWLCOND(IndicatorObsDA):
    """Class for observed water level conditional extremes.

    The backend indicator (wl_cond) is selected in relation to precipitation extremes.
    This class inherits logic to pick the best scipy distribution.

    The water levels are computed from a detrended time series of water levels, and adjusted to a reference period.
    To get results for another period, we need to adjust the water levels to the new period.
    """

    sl_mm_yr = param.Number(doc="Sea level rise rate (mm/y).")
    ref_period = param.Tuple(doc="Reference period for normalization.")
    data = param.Parameter(
        doc="DataArray time series of detrended observed conditional extremes, normalized to the reference period.",
        constant=True,
    )
    long_name = param.Parameter(
        "Water level extremes conditional on precipitation extremes"
    )

    @param.depends("data", watch=True, on_init=True)
    def _update_attrs(self) -> None:
        """Obtain attributes from the data."""
        for key in ["sl_mm_yr"]:
            setattr(self, key, self.data.attrs[key])
        setattr(self, "ref_period", tuple(self.data.attrs["ref_period"]))

    @param.depends("data", "ref_period", "sl_mm_yr", watch=True, on_init=True)
    def _update_ts(self) -> None:
        """Add trend to data."""
        self.ts = add_trend(
            self.data,
            slope=self.sl_mm_yr / 1000,
            midpoint=np.mean(self.ref_period),
        )

    def _sample(self, period: tuple) -> xr.DataArray:
        """Return the full detrended time series, renormalized over the desired period."""
        with xr.set_options(keep_attrs=True):
            return self.data + wl_norm(self.sl_mm_yr, period)

    def _dist_method(self, name: str, arg) -> xr.DataArray:
        """Call distrisbution method on the data."""
        with xr.set_options(keep_attrs=True):
            out = xc.indices.stats.dist_method(
                function=name, fit_params=self.dparams, arg=arg
            )
            out.name = name
        return out

    def fit(
        self, dist: str, period: tuple, size: int = None, iteration: int = 1
    ) -> xr.DataArray:
        """Fit the distribution to the data (normalized for period) over the full period."""
        # TODO v2: Account for parametric uncertainty in the observations.

        sample = self._sample(period)

        if iteration > 1:
            if size is None:
                size = len(sample)

            sample = bootstrap.resample(
                sample, size=size, iteration=iteration, replace=True
            )

        return xc.indices.stats.fit(sample, dist=dist, dim="time", method="ML")

    @param.depends("_update_attrs", "dist", "period", watch=True, on_init=True)
    def _update_params(self) -> None:
        """Update the distribution parameters."""
        super()._update_params()


class IndicatorObsPRPOT(IndicatorDA):
    """Class for observed precipitation peaks-over-threshold extremes (pr_pot).

    The precipitation extremes are fit with the generalized pareto distribution.
    """

    dist = param.Selector(default="genpareto", objects=["genpareto"])
    stn_thresh = param.Number(
        doc="Station threshold used to compute peaks-over-threshold values."
    )
    peaks_per_yr = param.Number(doc="Number of peaks per year.")
    long_name = param.Parameter("Precipitation peaks over threshold")

    @param.depends("data", watch=True, on_init=True)
    def _update_attrs(self) -> None:
        """Update attributes based on data attributes."""
        for key in ["stn_thresh", "peaks_per_yr"]:
            setattr(self, key, self.data.attrs[key])

    def _sample(self, period: tuple) -> xr.DataArray:
        """Return the full time series as-is (no slicing or normalization)."""
        return self.data

    def pdf(self, x) -> xr.DataArray:
        """Return the probability density function."""
        pdf = lambda x: xc.indices.stats.dist_method(
            function="pdf", fit_params=self.dparams, arg=x
        )
        sf = lambda x: xc.indices.stats.dist_method(
            function="sf", fit_params=self.dparams, arg=x
        )
        n = self.peaks_per_yr

        return n * (1 - sf(x)) ** (n - 1) * pdf(x)

    def _dist_method(self, name: str, arg) -> xr.DataArray:
        """Adjust argument or results for the fact that the pareto parameters describe events that may
        occur more than once per year on average. If there are 2 event per year, then the return period is halved.
        """
        func = lambda x: xc.indices.stats.dist_method(
            function=name, fit_params=self.dparams, arg=x
        )

        with xr.set_options(keep_attrs=True):
            out = scale_pareto(func, name, arg, self.peaks_per_yr)
            out.name = name
        return out

    def fit(
        self, dist: str, period: tuple, size: int = None, iteration: int = 1
    ) -> xr.DataArray:
        """Fit the distribution to the data over the full period."""
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


class IndicatorObsPRCOND(IndicatorObsDA):
    """Class for observed precipitation conditional extremes.

    The backend indicator (pr_pot) is selected in relation to water level extremes.
    This class inherits logic to pick the best scipy distribution.
    """

    long_name = param.Parameter(
        "Precipitation extremes conditional on water level extremes"
    )

    def _sample(self, period: tuple) -> xr.DataArray:
        """Return the full time series as-is (no slicing or normalization)."""
        return self.data

    def fit(
        self, dist: str, period: tuple, size: int = None, iteration: int = 1
    ) -> xr.DataArray:
        """Fit the distribution to the data over the full period."""
        # TODO v2: Account for parametric uncertainty in the observations.

        sample = self._sample(period)

        if iteration > 1:
            if size is None:
                size = len(sample)

            sample = bootstrap.resample(
                sample, size=size, iteration=iteration, replace=True
            )

        return xc.indices.stats.fit(sample, dist=dist, dim="time", method="ML")


class IndicatorSimWLCOND(IndicatorSimWL):
    """Class for simulated water level conditional extremes.

    A simulation delta is added to the water level extremes.
    IndicatorObsWLCOND is an input to this class.
    """

    dist = param.Selector(
        default=None,
        objects=[None] + scipy_dists,
        label="Distribution",
        doc="Statistical distribution",
        allow_refs=True,
    )
    obs = param.ClassSelector(class_=IndicatorObsWLCOND, doc="Observed indicator.")

    @param.depends("data", watch=True, on_init=True)
    def _update_attrs(self) -> None:
        """Update attributes based on data attributes."""
        for key in ["sl_mm_yr"]:
            setattr(self, key, self.data.attrs[key])
        setattr(self, "ref_period", tuple(self.data.attrs["ref_period"]))

    @param.depends("obs", watch=True, on_init=True)
    def _update_title(self):
        """Set default long_name from indicator."""
        if self.long_name == "" and self.obs:
            self.long_name = self.obs.long_name

    def pdf(self, x) -> xr.DataArray:
        """Return the probability density function."""
        return self._dist_method("pdf", x)

    def _dist_method(self, name: str, arg) -> xr.DataArray:
        """Apply the distribution method with mixing."""
        mix = XMixtureDistribution(params=self.dparams, weights=self.weights)
        return getattr(mix, name)(arg)

    @param.depends("dist", "period", "ts", watch=True, on_init=True)
    def _update_params(self) -> None:
        """Update the distribution parameters."""
        if self.dist is None:
            self.dist = self.obs.dist
        if self.dist is not None and self.ts is not None:
            self.dparams = self.fit(self.dist, self.period)


class IndicatorSimPRPOT(IndicatorDA):
    """Class for simulated precipitation peaks-over-threshold extremes.

    A simulation delta is added to the precipitation extremes.
    IndicatorObsPRPOT is an input to this class.
    """

    obs = param.ClassSelector(class_=IndicatorObsPRPOT, doc="Observed indicator.")
    data = param.Parameter(doc="Daily precipitation simulations.")
    model_weights = param.Parameter(doc="Weights for the models.", allow_refs=True)
    dist = param.Selector(default="genpareto", objects=["genpareto"], allow_refs=True)
    weights = param.Parameter(doc="Weights for the scenarios")
    num_samples = param.Number(150, doc="Number of samples to draw for the bootstrap.")
    sample_size = param.Number(30, doc="Size of the bootstrap samples.")
    level = param.Number(0.1, doc="Significance level for the KS test. Unused here.")

    @param.depends("obs", watch=True, on_init=True)
    def _update_title(self):
        """Set default long_name from indicator."""
        if self.long_name == "" and self.obs:
            self.long_name = self.obs.long_name

    def _sample(self, fut_period: tuple) -> xr.DataArray:
        """Return the precip simulation delta relative to the obs period."""
        uncropped_obs_period = (1960, 2014)
        ref = self.data.sel(time=self._slice(uncropped_obs_period)).quantile(
            0.95, dim="time"
        )
        fut = self.data.sel(time=self._slice(fut_period)).quantile(0.95, dim="time")
        pr_delta = fut / ref
        pr_delta.attrs = self.data.attrs.copy()
        pr_delta.attrs["long_name"] = (
            f"Daily precipitation flux at surface (delta between "
            f"{fut_period[0]}-{fut_period[1]} and {self.obs.period[0]}-{self.obs.period[1]})"
        )
        pr_delta.attrs["long_name_fr"] = (
            f"Flux de précipitation journalière à la surface (delta entre "
            f"{fut_period[0]}-{fut_period[1]} et {self.obs.period[0]}-{self.obs.period[1]})"
        )
        return pr_delta

    def fit(self, dist: str, period: tuple) -> xr.DataArray:
        """Fit the distribution to the data, creating a bootstrap sample combining parametric uncertainty for the
        observation data, and scenario/model uncertainty for the future data.
        """
        # Bootstrapped distribution parameters for the observations.
        dparams = self.obs.fit(
            dist, self.obs.period, size=self.sample_size, iteration=self.num_samples
        )
        pr_delta = self._sample(period).dropna(dim="realization", how="all")
        pr_delta = pr_delta.unstack("realization")

        # TODO - improve dimension handling for readability
        new_coords = pr_delta.coords.to_dataset().drop_dims("variant_label").coords
        new_dims = tuple(item for item in pr_delta.dims if item != "variant_label")
        new_coords["dparams"] = dparams.dparams.values
        new_coords["sample"] = dparams.coords["sample"]
        da = xr.DataArray(
            data=np.zeros(
                (len(new_coords["dparams"]),)
                + (pr_delta.source_id.size, pr_delta.experiment_id.size)
                + (len(new_coords["sample"]),)
            ),
            coords=new_coords,
            dims=("dparams",) + new_dims + ("sample",),
        )
        # da.loc["loc", ...] = pr_delta.squeeze("variant_label") * dparams.sel(
        #     dparams="loc"
        # )
        da.loc["loc", ...] = pr_delta.mean(dim="variant_label") * dparams.sel(
            dparams="loc"
        )
        for dparam in dparams.dparams.values:
            if dparam != "loc":
                da.loc[dparam, ...] = dparams.sel(dparams=dparam).values
        da.attrs = dparams.attrs
        return da

    def pdf(self, x) -> xr.DataArray:
        """Return the probability density function."""
        mix = XMixtureDistribution(params=self.dparams, weights=self.weights)
        n = self.obs.peaks_per_yr
        return n * (1 - mix.sf(x)) ** (n - 1) * mix.pdf(x)

    def _dist_method(self, name: str, arg) -> xr.DataArray:
        """Compute weighted statistics."""
        mix = XMixtureDistribution(params=self.dparams, weights=self.weights)
        func = getattr(mix, name)
        return scale_pareto(func, name, arg, self.obs.peaks_per_yr)

    @param.depends("period", "model_weights", watch=True, on_init=True)
    def _update_weights(self) -> None:
        """Compute the scenario and model weights over the period."""
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

        # Model weights
        model_w = model_weights_from_sherwood(
            self.dparams.source_id.values, method="L2Var", lambda_=0.5
        )

        # Combined weights
        w = scen_w * sample_w * model_w

        # Account for misaligned indices
        if self.dparams.sel(dparams="loc").isnull().sum() != 0:
            aligned_w = w.transpose(*self.dparams.sel(dparams="loc").dims)
            masked_w = aligned_w.where(
                ~self.dparams.sel(dparams="loc").isnull(), other=np.nan
            )
            missing_w_ratio = 1 / ((masked_w).sum())
            w = masked_w * missing_w_ratio
            w = w.fillna(0)

        self.weights = w

    # Placeholder reminder.
    # def experiment_percentiles(self, per) -> xr.Dataset:
    #     return


class IndicatorSimPRCOND(IndicatorSimPRPOT):
    """Class for simulated precipitation conditional extremes.

    A simulation delta is added to the precipitation extremes.
    IndicatorObsPRCOND is an input to this class.
    """

    obs = param.ClassSelector(class_=IndicatorObsPRCOND, doc="Observed indicator.")
    dist = param.Selector(
        default=None,
        objects=[None] + scipy_dists,
        label="Distribution",
        doc="Statistical distribution",
        allow_refs=True,
    )
    wl_pot = param.Parameter(
        doc="Water level peaks-over-threshold extremes use to calculate precipitation simulation delta."
    )

    def __init__(self, wl_pot: xr.DataArray, **kwargs):
        """Initialize the indicator."""
        self.wl_pot = wl_pot
        super().__init__(**kwargs)

    @param.depends("obs", watch=True, on_init=True)
    def _update_title(self):
        """Set default long_name from indicator."""
        if self.long_name == "" and self.obs:
            self.long_name = self.obs.long_name

    def _sample(self, period: tuple) -> xr.DataArray:
        """Calculate precipitation simulation delta relative to the reference period.

        The delta is calculated as a ratio between subsets of the reference and future periods.
        These subsets, representing conditional extremes, are identified in the reference period
        based on water level extremes and applied to the future period using the reference period quantiles.
        """
        uncropped_obs_period = (
            1960,
            2014,
        )  # Alternatively could go to present-day as for the wl
        pr_ref = self.data.sel(time=self._slice(uncropped_obs_period))
        pr_fut = self.data.sel(time=self._slice(period))

        # TODO - replace with xr.apply_ufunc for efficiency
        deltas = []
        for i in range(len(pr_ref.realization)):
            pr_ref_real = pr_ref.isel(realization=i)
            pr_fut_real = pr_fut.isel(realization=i)

            _, pr_cond = matching_events(self.wl_pot, pr_ref_real)
            q = define_q(pr_ref_real.values, pr_cond)
            pr_ref_subset = np.quantile(pr_ref_real.values, q)
            pr_fut_subset = np.quantile(pr_fut_real.values, q)

            delta_ratio = np.median(pr_fut_subset) / np.median(pr_ref_subset)
            deltas.append(delta_ratio)

        deltas_array = xr.DataArray(
            data=deltas,
            dims=["realization"],
            coords={
                "realization": pr_ref.realization,
            },
            attrs=pr_ref.attrs,
        )

        return deltas_array

    def pdf(self, x) -> xr.DataArray:
        """Return the probability density function."""
        return self._dist_method("pdf", x)

    def _dist_method(self, name: str, arg) -> xr.DataArray:
        """Apply the distribution method with mixing."""
        mix = XMixtureDistribution(params=self.dparams, weights=self.weights)
        return getattr(mix, name)(arg)

    @param.depends("dist", "period", "ts", watch=True, on_init=True)
    def _update_params(self) -> None:
        """Update the distribution parameters."""
        if self.dist is None:
            self.dist = self.obs.dist
        if self.dist is not None and self.ts is not None:
            self.dparams = self.fit(self.dist, self.period)


class IndicatorObsJP(IndicatorObsDA):
    """Joint probabiliy (copula) analysis for observations.

    Distribution selection logic is inherited from IndicatorObsDA and applied to copula selection.
    Inputs to this class are either:

        1. IndicatorObsWL and IndicatorObsPRCOND (indicator wl_prcond), or
        2. IndicatorObsPRPOT and IndicatorObsWLCOND (indicator pr_wlcond).
    """

    dist = param.Selector(
        default=None,
        objects=[None] + copulas,
        label="Distribution",
        doc="Statistical distribution",
        allow_refs=True,
    )
    obs_pot = param.ClassSelector(
        class_=(IndicatorObsWL, IndicatorObsPRPOT),
        doc="Peaks-over-threshold marginal (observations)",
    )
    obs_cond = param.ClassSelector(
        class_=(IndicatorObsWLCOND, IndicatorObsPRCOND),
        doc="Conditional marginal (observations)",
    )
    data = param.Parameter(
        doc="Pseudo-observations from the marginals (2D).", constant=False
    )

    level = param.Number(0.05, doc="Significance level for the Sn test.")

    def __init__(
        self, obs_pot: xr.DataArray, obs_cond: xr.DataArray, period: tuple, **kwargs
    ):
        """Initialize the indicator with the marginals"""
        self.obs_pot = obs_pot
        self.obs_cond = obs_cond
        self.period = period
        self.dist = kwargs.get("dist", None)
        kwargs.update({"obs_pot": obs_pot, "obs_cond": obs_cond, "period": period})

        super().__init__(**kwargs)

        self._update_title()
        self._update_ts()
        self._update_marginals()

    def _update_title(self) -> None:
        """Get long_name from data if not explicitly set."""
        if self.long_name == "":
            if self.obs_pot.data.name == "wl_pot":
                self.long_name = "Joint water level peaks over threshold and conditional precipitation extremes"
                self.data.attrs["long_name"] = (
                    "Joint water level peaks over threshold and conditional precipitation extremes"
                )
                self.data.attrs["long_name_fr"] = (
                    "Pics conjoints de niveau d'eau au-dessus du seuil et de précipitations extrêmes conditionnelles"
                )
                self.data.attrs["description"] = (
                    "Joint water level peaks over threshold and conditional precipitation extremes"
                )
                self.data.attrs["description_fr"] = (
                    "Pics conjoints de niveau d'eau au-dessus du seuil et de précipitations extrêmes conditionnelles"
                )
            elif self.obs_pot.data.name == "pr_pot":
                self.long_name = "Joint precipitation peaks over threshold and conditional water extremes"
                self.data.attrs["long_name"] = (
                    "Joint precipitation peaks over threshold and conditional water extremes"
                )
                self.data.attrs["long_name_fr"] = (
                    "Pics conjoints de précipitations au-dessus du seuil et d'extrêmes de niveau d'eau conditionnels"
                )
                self.data.attrs["description"] = (
                    "Joint precipitation peaks over threshold and conditional water extremes"
                )
                self.data.attrs["description_fr"] = (
                    "Pics conjoints de précipitations au-dessus du seuil et d'extrêmes de niveau d'eau conditionnels"
                )

    def _update_ts(self) -> None:
        """Set the time series to be displayed."""
        self.ts = self.data

    def _sample(self, period) -> None:
        """Get pseudo-observations from the marginals and combine into one xarray (no slicing or normalization)."""
        combined = np.concatenate(
            [
                self.obs_pot.data.values.reshape(-1, 1),
                self.obs_cond.data.values.reshape(-1, 1),
            ],
            axis=1,
        )
        pobs_data = pobs(combined)
        data = xr.DataArray(
            pobs_data,
            dims=["time", "marg"],
            coords={"time": self.obs_pot.data.time, "marg": ["pot", "cond"]},
        )
        if self.obs_pot.name == "wl_pot":
            data.attrs["name"] = "wl_prcond"
            data.attrs["title"] = "wl_prcond"
            data.attrs["wl_pot_attrs"] = self.obs_pot.data.attrs
            data.attrs["pr_cond_attrs"] = self.obs_cond.data.attrs
        elif self.obs_pot.name == "pr_pot":
            data.attrs["name"] = "pr_wlcond"
            data.attrs["title"] = "pr_wlcond"
            data.attrs["pr_pot_attrs"] = self.obs_pot.data.attrs
            data.attrs["wl_cond_attrs"] = self.obs_cond.data.attrs
        self.data = data
        return data

    def fit(self, dist: str, period: tuple) -> xr.DataArray:
        """Fit the distribution to the data."""
        sample = self._sample(period)
        if dist == "indep":
            dparams = xr.DataArray(
                [np.nan], dims=["dparams"], coords={"dparams": ["param"]}
            )
        else:
            cop = copulae_copula(dist)
            cop.fit(sample.data, method="ml", verbose=0, optim_options=optim_options)

            if dist == "student":
                dparams = xr.DataArray(
                    [cop.params[0], cop.params[1][0]],
                    dims=["dparams"],
                    coords={"dparams": ["df", "rho"]},
                )
            else:
                dparams = xr.DataArray(
                    cop.params, dims=["dparams"], coords={"dparams": "param"}
                )

        if not check_param(dparams, dist):
            raise ValueError("Invalid parameter for ", dist)

        dparams.name = self.data.name
        attrs = {
            "title": self.data.name,
            "original_long_name": self.data.name,
            "long_name": f"Parameters of the {dist} copula",
            "long_name_fr": f"Paramètres de la copule {dist}",
            "description": f"Parameters of the {dist} copula",
            "description_fr": f"Paramètres de la copule {dist}",
            "method": "ML",
            "estimator": "Maximum likelihood",
            "scipy_dist": {dist},
            "units": "",
            "history": (
                f'{datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")} '
                "fit: Estimate copula parameters by maximum likelihood method "
                "along dimension time. - copulae version: 0.7.9"
            ),
        }
        dparams.attrs = attrs
        return dparams

    def _dist_method(self, name: str, arg: Sequence[float]) -> xr.DataArray:
        """Apply the distribution method separately to each marginal and the copula.

        Pass two values as `arg`, one for each marginal (e.g., thresholds in mm or m CGVD2013).
        """
        # Convert marginal argument to probability and undo the effect of scale_pareto.
        p1 = self.obs_pot.sf(arg[0]).item() / self.obs_pot.peaks_per_yr
        p2 = self.obs_cond.sf(arg[1]).item()
        cop = ot_copula(self.dist, self.dparams)

        # Apply scale_pareto to copula output.
        if name == "sf":
            return xr.DataArray(
                scale_pareto(
                    cop.computeSurvivalFunction,
                    name,
                    [1 - p1, 1 - p2],
                    self.obs_pot.peaks_per_yr,
                ),
                name=name,
            )
        elif name == "cdf":
            # Keeping this separate as 'cdf' is not implemented in scale_pareto
            return xr.DataArray(
                cop.computeCDF([1 - p1, 1 - p2]) * self.obs_pot.peaks_per_yr, name=name
            )

    def pdf(self, x):
        """Return the probability density function."""
        raise NotImplementedError("Method not supported in joint probability class.")

    def ppf(self, value) -> float:
        """Percent Point Function, inverse of CDF."""
        raise NotImplementedError("Method not supported in joint probability class.")

    def isf(self, value):
        """Inverse survival function."""
        raise NotImplementedError("Method not supported in joint probability class.")

    def _ll(self, params, sample) -> xr.DataArray:
        """Log-likelihood."""
        return NotImplementedError("Method not supported in joint probability class.")

    def _bic(self, dist: str, period: tuple) -> xr.DataArray:
        """Return the Bayesian Information Criterion."""
        sample = self._sample(period)
        dparams = self.fit(dist, period)
        bic = ot_bic(sample, dist, dparams)
        attrs = {
            "long_name": "Bayesian Information Criterion",
            "description": "Calculated using Openturns library",
            "history": update_history(
                "BIC", new_name="bic", parameters=dparams, sample=sample
            ),  # TODO - this function adds xclim to the history (incorrect)
            "scipy_dist": dist,
            "period": period,
        }
        out = xr.DataArray(bic, attrs=attrs)
        out.name = self.data.name
        return out

    def _aic(self, dist: str, period: tuple) -> xr.DataArray:
        """Return the Akaike Information Criterion."""
        sample = self._sample(period)
        dparams = self.fit(dist, period)
        aic = ot_aic(sample, dist, dparams)
        attrs = {
            "long_name": "Akaike Information Criterion",
            "description": "Calculated using Openturns library",
            "history": update_history(
                "AIC", new_name="aic", parameters=dparams, sample=sample
            ),  # TODO - this is adding xclim to the history (incorrect)
            "scipy_dist": dist,
            "period": period,
        }
        out = xr.DataArray(aic, attrs=attrs)
        out.name = self.data.name
        return out

    @param.depends("period", "metric", watch=True)
    def _update_metrics(self) -> None:
        """Return the metric values for all distributions.

        Parameters
        ----------
        metric : {'aic', 'bic'}
            Information criterion.
        """
        if self.metric not in ["aic", "bic"]:
            raise ValueError(f"Unknown metric {self.metric}")

        # Reset values
        self.metrics = {"aic": None, "bic": None}

        # Update metrics values for given metric
        self.metrics[self.metric] = {
            dist: getattr(self, f"_{self.metric}")(dist, self.period)
            for dist in copulas  # note: copulas not scipy_dists
        }
        self.param.trigger("metrics")

    # def _sample(self, period: tuple) -> xr.DataArray:
    #     """Return the pseudodata as-is (no slicing or normalization)."""
    #     return self.data

    @param.depends("period", "metric", watch=True)
    def _update_marginals(self) -> None:
        """Update marginals (e.g., normalization) with new period."""
        self.obs_pot.period = self.period
        self.obs_cond.period = self.period

    def _Sn(self) -> float:
        """Get Sn and p_val goodness-of-fit copula test (Genest et al. 2009) from copulae package."""
        # Placeholder.
        Sn_p = self.level
        return Sn_p

    @param.depends("dist", "period", watch=True, on_init=True)
    def _update_params(self) -> None:
        """Update the distribution parameters."""
        if self.dist is None:
            self.dist = self.best_dist()
        Sn_p = self._Sn()
        if Sn_p >= self.level:
            self.dparams = self.fit(self.dist, self.period)
        else:
            raise ValueError(
                "The copula with lowest bic has been rejected by Sn goodness-of-fit test."
            )

    # Placeholder reminder.
    # def _update_ts(self):
    #     self.ts = self.data


class IndicatorSimJP(param.Parameterized):
    """Joint probabiliy (copula) analysis for simulations.

    Inputs to this class are IndicatorObsJP and either:
        (1) IndicatorSimWL and IndicatorSimPRCOND (indicator wl_prcond), or
        (2) IndicatorSimPRPOT and IndicatorSimWLCOND (indicator pr_wlcond).
    """

    dist = param.Selector(
        objects=copulas,
        label="Distribution",
        doc="Statistical distribution",
        allow_refs=True,
    )
    sim_pot = param.ClassSelector(
        class_=(IndicatorSimWL, IndicatorSimPRPOT),
        doc="Peaks-over-threshold marginal (simulations)",
    )
    sim_cond = param.ClassSelector(
        class_=(IndicatorSimWLCOND, IndicatorSimPRCOND),
        doc="Conditional marginal (simulations)",
    )
    obs_cop = param.ClassSelector(
        class_=IndicatorObsJP, doc="Copula derived from observations"
    )
    data = param.Parameter(
        doc="Pseudo-observations from the marginals (2D).", constant=False
    )
    period = param.Range(
        allow_refs=True, doc="Period for statistical analysis", allow_None=False
    )
    title = param.String("", doc="Indicator name")
    long_name = param.String("", doc="Short indicator name")
    description = param.String("", doc="Indicator description")

    def __init__(
        self,
        sim_pot: xr.DataArray,
        sim_cond: xr.DataArray,
        obs_cop: xr.DataArray,
        period: tuple,
        **kwargs,
    ):
        """Initialize the indicator with the marginals."""
        super().__init__(**kwargs)
        self.sim_pot = sim_pot
        self.sim_cond = sim_cond
        self.period = period
        self.dist, self.dparams, self.data = obs_cop.dist, obs_cop.dparams, obs_cop.data
        self._update_title()
        self._update_marginals()

    def _update_title(self) -> None:
        """Get long_name from data if not explicitly set."""
        if self.long_name == "":
            if self.sim_pot.obs.data.name == "wl_pot":
                self.long_name = "Joint water level peaks over threshold and conditional precipitation extremes"
                self.data.attrs["long_name"] = (
                    "Joint water level peaks over threshold and conditional precipitation extremes"
                )
                self.data.attrs["long_name_fr"] = (
                    "Pics conjoints de niveau d'eau au-dessus du seuil et de précipitations extrêmes conditionnelles"
                )
                self.data.attrs["description"] = (
                    "Joint water level peaks over threshold and conditional precipitation extremes"
                )
                self.data.attrs["description_fr"] = (
                    "Pics conjoints de niveau d'eau au-dessus du seuil et de précipitations extrêmes conditionnelles"
                )
            elif self.sim_pot.obs.data.name == "pr_pot":
                self.long_name = "Joint precipitation peaks over threshold and conditional water extremes"
                self.data.attrs["long_name"] = (
                    "Joint precipitation peaks over threshold and conditional water extremes"
                )
                self.data.attrs["long_name_fr"] = (
                    "Pics conjoints de précipitations au-dessus du seuil et d'extrêmes de niveau d'eau conditionnels"
                )
                self.data.attrs["description"] = (
                    "Joint precipitation peaks over threshold and conditional water extremes"
                )
                self.data.attrs["description_fr"] = (
                    "Pics conjoints de précipitations au-dessus du seuil et d'extrêmes de niveau d'eau conditionnels"
                )

    def _dist_method(self, name: str, arg: Sequence[float]) -> xr.DataArray:
        """Apply the distribution method separately to each marginal and the copula.

        Pass two values as `arg`, one for each marginal (e.g., thresholds in mm or m CGVD2013).
        """
        # Convert marginal argument to probability and undo the effect of scale_pareto.
        p1 = self.sim_pot.sf(arg[0]).item() / self.sim_pot.obs.peaks_per_yr
        p2 = self.sim_cond.sf(arg[1]).item()
        cop = ot_copula(self.dist, self.dparams)

        map = {"sf": cop.computeSurvivalFunction, "cdf": cop.computeCDF}

        # Apply scale_paereto to copula output.
        if name == "sf":
            return xr.DataArray(
                scale_pareto(
                    map[name], name, [1 - p1, 1 - p2], self.sim_pot.obs.peaks_per_yr
                ),
                name=name,
            )
        elif name == "cdf":
            # Keeping this separate as 'cdf' is not implemented in scale_pareto
            return xr.DataArray(
                cop.computeCDF([1 - p1, 1 - p2]) * self.sim_pot.obs.peaks_per_yr,
                name=name,
            )

    def sf(self, value) -> xr.DataArray:
        """Survival function (1-CDF), or probability of exceedance."""
        return self._dist_method("sf", value)

    def cdf(self, value) -> xr.DataArray:
        """Cumulative Distribution Function (CDF)."""
        return self._dist_method("cdf", value)

    def pdf(self, x):
        """Return the probability density function."""
        raise NotImplementedError("Method not supported in joint probability class.")

    def ppf(self, value) -> float:
        """Percent Point Function, inverse of CDF."""
        raise NotImplementedError("Method not supported in joint probability class.")

    def isf(self, value):
        """Inverse survival function."""
        raise NotImplementedError("Method not supported in joint probability class.")

    def _ll(self, params, sample) -> xr.DataArray:
        """Log-likelihood."""
        return NotImplementedError("Method not supported in joint probability class.")

    @param.depends("period", watch=True)
    def _update_marginals(self) -> None:
        """Update marginals with new period."""
        self.sim_pot.period = self.period
        self.sim_cond.period = self.period

    # Placeholder reminder.
    # def _update_ts(self):
    #     self.ts = self.data
    # def experiment_percentiles(self, per) -> xr.Dataset:
    #     return
