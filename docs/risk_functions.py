from xclim.core.formatting import update_history
from portail_ing.risk.mixture import parametric_logpdf as logpdf
import xarray as xr
import numpy as np
from scipy import stats
import xclim as xc
import pandas as pd
from portail_ing.risk.priors import (
    members,
    model_weights_from_sherwood,
    scenario_weights_from_iams,
)
from portail_ing.risk.xmixture import XMixtureDistribution
from portail_ing.frontend.parameters import pievc

# This code is based on the portail_ing.front.parameters without GUI parts 

class metrics_da:
    
    """Class to fit and test metrics distribution for observations.
    
    Parameters
    ----------
    da : xr.DataArary
        Observed indicator DataArray.
    scipy_dist: list of strings
        Scipy dist to be testted.
    period: list of strings
        Period for analysis (ex: ['1980', '2010'])
    metric: str
        'aic' or 'bic' to find the best fit for distribution. 
    """
    def __init__(self, da, scipy_dists, period, metric):
        
        self.scipy_dists = scipy_dists
        self.period = period
        self.sample = da.sel(time=slice(*self.period))

        self.metric = metric
        if self.metric not in ['aic', 'bic']:
            raise ValueError('Can only use aic or bic for metric')
        self.metrics = {'aic': None, 'bic': None}
        self.metrics[self.metric] = {
            dist: getattr(self, f"_{self.metric}")(dist)
            for dist in self.scipy_dists
        }
        
    def _ll(self, params) -> xr.DataArray:
        """Return the log-likelihood of the distribution."""
        return logpdf(params, v=self.sample).sum(dim="logpdf")
    
    def fit(self, dist) -> xr.DataArray:
        """Fit the distribution to the data."""
        return xc.indices.stats.fit(self.sample, dist=dist, dim="time", method="ML")
    
    def _aic(self, dist) -> xr.DataArray:
        dparams = self.fit(dist)
        ll = self._ll(dparams)
        out = 2 * len(dparams) - 2 * ll
        out.attrs = {
            "long_name": "Akaike Information Criterion",
            "description": "AIC = 2 k - 2 log(L)",
            "history": update_history(
                "AIC", new_name="aic", 
                parameters=dparams, 
                sample=self.sample,
            ),
            "scipy_dist": dist,
            "period": self.period,
        }
        return out
        
    def _bic(self, dist: str) -> xr.DataArray:
        """Return the Bayesian Information Criterion.

        BIC = log(n) k - 2 log(L)
        """
        dparams = self.fit(dist)
        ll = self._ll(dparams)
        out = np.log(len(self.sample)) * len(dparams) - 2 * ll
        out.attrs = {
            "long_name": "Bayesian Information Criterion",
            "description": "BIC = log(n) k - 2 log(L)",
            "history": update_history(
                "BIC", 
                new_name="bic", 
                parameters=dparams, 
                sample=self.sample,
            ),
            "scipy_dist": dist,
            "period": self.period,
        }
        return out

    @property
    def metrics_da(self) -> xr.DataArray:
        """Return metrics DataArray."""
        if self.metrics[self.metric] is not None:
            vals = [
                val.expand_dims(scipy_dist=[dist])
                for dist, val in self.metrics[self.metric].items()
            ]
            out = xr.concat(vals, dim="scipy_dist")
            out.attrs.pop("scipy_dist")
            return out

    def best_dist(self) -> str:
        """Return the distribution with the best metric value.

        Parameters
        ----------
        metric: {'aic', 'bic'}
          Information criterion that we seek to minimize.
        """
        return self.metrics_da.idxmin("scipy_dist").item()
    
    def _dist_method(self, name, arg: xr.DataArray | float):
        self.dparams = self.fit(self.best_dist())
        with xr.set_options(keep_attrs=True):
            out = xc.indices.stats.dist_method(
                function=name, fit_params=self.dparams, arg=arg
            )
        out.name = name
        return out

    def pdf(self, x):
        """Return the probability density function."""
        return self._dist_method("pdf", x)

    def cdf(self, value) -> float:
        """Cumulative Distribution Function (CDF)."""
        return self._dist_method("cdf", value)

    def ppf(self, value) -> float:
        """Percent Point Function, inverse of CDF."""
        return self._dist_method("ppf", value)

    def sf(self, value) -> float:
        """Survival function (1-CDF), or probability of exceedance."""
        return self._dist_method("sf", value)

    def isf(self, value):
        """Inverse survival function."""
        return self._dist_method("isf", value)

# test KS (significance)
def ks(obs, sim, period, level, rdim) -> xr.DataArray:
    """Kolmogorov-Smirnov test between the observed and simulated data.

    The null hypothesis is that the two distributions are identical.
    If the p-value < 0.05, we reject this hypothesis and return a weight of 0. otherwise we consider the
    distributions similar and return 1.

    Parameters
    ----------
    level : float
        The significance level for the test. Increase the value to include more models.

    Returns
    -------
    xr.DataArray
        1 if both distributions are similar over the reference period, 0 otherwise.
    """
    
    obs = obs.sel(time=slice(period[0], period[1]))
    ref = sim.sel(time=slice(period[0], period[1]))
    
    # Do we want to stack by source_id and member or only source?
    #ref = ref.stack(tr=["time", "variant_label"]).isel(experiment_id=0)

    # We assume that over the reference period, the values are the same for all the experiments.
    def func_ks(r):
        if (~np.isnan(r)).sum() < 10:
            return 0

        return stats.ks_2samp(
            r,
            obs,
            alternative="two-sided",
            method="auto",
            axis=0,
            nan_policy="omit",
            keepdims=False,
        ).pvalue

    # func = lambda x: 1

    p = xr.apply_ufunc(
        func_ks,
        ref,
        input_core_dims=[[rdim]],
        vectorize=True,
        dask="parallelized",
    )

    return p > level


def scenario_weights(period):
    """GES scnenario weights (experiment_id) based on IAM likelihood"""
    scen_weights = scenario_weights_from_iams()
    w = scen_weights.sel(time=slice(*period)).mean(dim="time")
    if w.isnull().all():
            w = w.fillna(0.25)
    return w

def model_weights(ks_da, dim='realization', method ='L2Var', lambda_=0.5):
    """Model (source_id) weights based on ECS (Zelinka)"""
    
    ok = ks_da.where(ks_da).dropna(dim).groupby('source_id')
    
    #check if enough models with ks test    
    if len(ok)<2:
       raise ValueError("Not enough models to compute weights.")
    
    models  = list(ok.groups.keys())
    w = model_weights_from_sherwood(
            models, method=method, lambda_=lambda_
        )
    return w 

def combined_weights(da, ks_da, scen_weights, model_weights, dim='realization', w_w = True):
    """Combine KS, experiment and source weights for all realization in da"""
    
    variant_label_counts = da.sel(realization=ks_da.dropna(dim=dim)[dim].values).isel(time=0).drop_vars('time').groupby("source_id").count(dim=dim)
    varl_weights = 1 / variant_label_counts
    
    weights = {}
    for rea in da[dim].values:
        w1 = scen_weights.sel(experiment_id=rea.split('_')[1])
        w2 = model_weights.sel(source_id=rea.split('_')[0])
        w3 = varl_weights.sel(source_id=rea.split('_')[0])
        weights[rea] =  w1 * w2 * w3
    
    concatenated = xr.concat(weights.values(), dim=dim)
    concatenated[dim] = list(weights.keys())
    
    concatenated = concatenated.drop_vars(['source_id', 'ecs'])
        
    if w_w == False:
        concatenated[:] = 1
        
    return concatenated / concatenated.sum(dim='realization') 

class mixture:    
    """Class to create mixture and add weights.
    
    Parameters
    -----------
    da: xr.DataArray
        Simulations indicator DataArray. Needs to have the realizations dimension.
    dist: str
        Scipy distribution to be fitted to the simulations.
    period: list of str
        Period for the mixture.
    ks_da: xr.DataArray
        Kolmogorov-Smirnov two sample test.
    w_w: bool
        True: with weights; ks_da, models and scenarios weights will be combined and added to the mxiture
        False: no weights; all simulations are equal 
    
    """
     
    def __init__(self, da, dist, period, ks_da, w_w=True):        
        self.dist = dist
        self.period = period
        if w_w:
            self.sample = da.sel(realization=ks_da.dropna(dim='realization')['realization'].values).sel(time=slice(*period))
            self.weights = combined_weights(
                self.sample,
                ks_da,
                scenario_weights(self.period),
                model_weights(ks_da)
            )
        else:
            self.sample = da.sel(time=slice(*period))
            self.weights = self.weights = combined_weights(
                self.sample,
                ks_da,
                scenario_weights(self.period),
                model_weights(ks_da),
                w_w=False,
            )
        
    def fit(self) -> xr.DataArray:
        """Fit the distribution to the data."""
        return xc.indices.stats.fit(self.sample, dist=self.dist, dim="time", method="ML")

    def _dist_method(self, name, arg: xr.DataArray | float):
        self.dparams = self.fit()
        mix = XMixtureDistribution(params=self.dparams, weights=self.weights)
        return getattr(mix, name)(arg)

    def pdf(self, x):
        """Return the probability density function."""
        return self._dist_method("pdf", x)

    def cdf(self, value) -> float:
        """Cumulative Distribution Function (CDF)."""
        return self._dist_method("cdf", value)

    def ppf(self, value) -> float:
        """Percent Point Function, inverse of CDF."""
        return self._dist_method("ppf", value)

    def sf(self, value) -> float:
        """Survival function (1-CDF), or probability of exceedance."""
        return self._dist_method("sf", value)

    def isf(self, value):
        """Inverse survival function."""
        return self._dist_method("isf", value)
    
class exceedance(): # j'ai enlever la section élément à risque... devrait peut être ajouter
    """ Class Class for hazards thresholds.

    Facilitate going from values to return periods and vice-versa.

    `sf` stands for survival function (1-CDF)
    
    Parameters
    ----------
    obs: class
        metrics_da class for reference period.
    ref: class 
        mixture class for reference period.
    fut: class  
        mixture class for futur period.
    input: str
        'X' or 'T'. If 'T', value will be a return period and 'X' a threshold.
    input_value: float
        Value associated to type. 
    locale: str
        'en' or 'fr'. Language to be used for likelihood talbe.
    """
    _keys = [
        # "long_name",
        "xid",
        #"descr",
        "obs_t",
        "value",
        "obs_sf",
        "ref_sf",
        "fut_sf",
        "ratio",
    ]
    
    _doc = {
        "xid": {"en": "Climate hazard", "fr": "Aléa climatique"},
        #"descr": {
        #    "en": "Component affected by hazard",
        #    "fr": "La composante affectée par l'aléa",
        #},
        "obs_t": {
            "en": "Return period during the reference period",
            "fr": "Temps de retour pendant la période de référence",
        },
        "value": {"en": "Climate threshold value", "fr": "Seuil climatique"},
        "obs_sf": {
            "en": "Exceedance probability during the reference period",
            "fr": "Probabilité de dépassement pendant la période de référence",
        },
        "ref_sf": {
            "en": "Exceedance probability during the reference period",
            "fr": "Probabilité de dépassement pendant la période de référence",
        },
        "fut_sf": {
            "en": "Exceedance probability during the future period",
            "fr": "Probabilité de dépassement pendant la période future",
        },
        "ratio": {
            "en": "Ratio of future exceedance probability vs reference",
            "fr": "Ratio de probabilité de dépassement future vs référence",
        },
    }
        
    def __init__(self, obs, ref, fut, input, input_value, locale='fr'):
        self.obs = obs
        self.ref = ref
        self.fut = fut
        self.input = input
        self.locale = locale
        if input == 'X':
            self.value = input_value
        elif input == 'T':
            self.obs_t = input_value
        if locale=='fr':
            self.xid = obs.sample.attrs['description_fr']
        elif locale=='en':
            self.xid = obs.sample.attrs['description']
        
        self._update_from_value()
        self._update_from_obs_t()
        self._update_sim()
        
    def _update_from_value(self):
        """Compute return period from value."""
        if self.input == "X":
            self.obs_sf = self.obs.sf(self.value).item()
            self.obs_t = 1 / self.obs_sf

    def _update_from_obs_t(self):
        """Compute value from return period."""
        if self.input == "T":
            self.obs_sf = 1 / self.obs_t
            self.value = self.obs.isf(self.obs_sf).item()

    def _update_sim(self):
        """Compute simulated exceedance probabilities and ratio."""
        if self.value is not None:
            self.ref_sf = self.ref.sf(self.value).item()
            self.fut_sf = self.fut.sf(self.value).item()
            self.ratio = self.fut_sf / self.ref_sf - 1

    def values(self) -> pd.Series:
        """Return values Series."""
        values = [getattr(self, key, None) for key in self._keys]
        return pd.Series(values, index=self._keys)

    def titles(self) -> dict:
        """Return labels Series."""
        return {key: getattr(key, "label") for key in self._keys}

    def docs(self):
        return {self._doc[key][self.locale]: self.values().loc[[key]].item() for key in self._keys}   
    
    def pievc(self):
        if self.locale=='fr':
            k = ['Score historique', 'Score futur']
        elif self.locale=='en':
            k = ['Historical Score', 'Futur Score']
        return dict(zip(k, [pievc(self.ref_sf).item(), pievc(self.fut_sf).item()]))
    

    
    
    
    

