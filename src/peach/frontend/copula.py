"""
Interface for copulas with `copulae` and `OpenTURNS`.

Key Functions:
- `copulae_copula`: Retrieve copula objects from `copulae`.
- `ot_marginal` and `ot_copula`: Retrieve marginal and copula objects from `OpenTURNS`.
- `ot_aic`/`ot_bic`: Calculate with `OpenTURNS`.
"""

import copulae
import numpy as np
import openturns as ot
import xarray as xr

optim_options = {
    "tol": 1e-6,
    "options": {
        "maxiter": 1000,
    },
}

copulae_map = {
    "gaussian": copulae.GaussianCopula(dim=2),
    "student": copulae.StudentCopula(dim=2),
    "clayton": copulae.ClaytonCopula(dim=2),
    "frank": copulae.FrankCopula(dim=2),
    "gumbel": copulae.GumbelCopula(dim=2),
    "indep": copulae.IndepCopula(dim=2),
}
ot_map = {
    "gaussian": ot.NormalCopula,
    "student": ot.StudentCopula,
    "clayton": ot.ClaytonCopula,
    "frank": ot.FrankCopula,
    "gumbel": ot.GumbelCopula,
    "indep": ot.IndependentCopula,
}

ot_map_marginals = {
    "norm": ot.Normal,
    "t": ot.Student,
    "gamma": ot.Gamma,
    "genextreme": ot.GeneralizedExtremeValue,
    "genpareto": ot.GeneralizedPareto,
    "lognorm": ot.LogNormal,
    "uniform": ot.Uniform,
}


def pobs(data: np.ndarray):
    """Convert data into pseudo-observations."""
    return copulae.core.pseudo_obs(data, ties="average")


def copulae_copula(dist: str):
    """Obtain copula object from the copulae package."""
    if dist not in copulae_map:
        raise NotImplementedError(f"Copula family '{dist}' is not implemented.")
    return copulae_map[dist]


def check_param(dparams: xr.DataArray, family: str) -> bool:
    """Double check copula parameters using theoretical bounds."""
    if family == "gaussian":
        return -1 < dparams.item() < 1
    elif family == "student":
        return (
            dparams.sel(dparams="df").item() > 0
            and -1 < dparams.sel(dparams="rho").item() < 1
        )
    elif family == "clayton":
        return dparams.item() >= -1 and dparams.item() != 0
    elif family == "frank":
        return dparams.item() != 0
    elif family == "gumbel":
        return dparams.item() >= 1
    elif family == "indep":
        return dparams.isnull().all()
    else:
        raise ValueError(f"Copula family '{family}' is not implemented.")


def ot_matrix(corr: float, dim: int = 2):
    """Obtain OpenTURNS correlation matrix."""
    matrix = ot.CorrelationMatrix(dim)
    matrix[0, 1] = corr
    matrix[1, 0] = corr
    return matrix


def ot_marginal(dist: str, dparams: xr.DataArray):
    """Obtain OpenTURNS univariate distribution object."""
    if dist not in ot_map_marginals:
        raise NotImplementedError(f"Marginal distribution '{dist}' is not implemented.")
    params = dparams.values  # a/c, loc, scale
    if dist == "norm":
        return ot_map_marginals[dist](params[0], params[1])
    elif dist == "t":
        return ot_map_marginals[dist](params[0], params[1], params[2])
    elif dist == "gamma":
        return ot_map_marginals[dist](params[0], 1 / params[2], params[1])
    elif dist == "genextreme":
        return ot_map_marginals[dist](params[1], params[2], -params[0])
    elif dist == "genpareto":
        return ot_map_marginals[dist](params[2], params[0], params[1])
    elif dist == "lognorm":
        muLog = np.log(params[2])
        sigmaLog = params[0]
        gamma = params[1]
        return ot_map_marginals[dist](muLog, sigmaLog, gamma)
    elif dist == "uniform":
        return ot_map_marginals[dist](params[0], params[0] + params[1])
    return


def ot_copula(dist: str, dparams: xr.DataArray):
    """Obtain OpenTURNS copula object."""
    if dist not in ot_map:
        raise NotImplementedError(f"Copula family '{dist}' is not implemented.")
    if dist == "student":
        matrix = ot_matrix(dparams.sel(dparams="rho").item())
        return ot_map[dist](dparams.sel(dparams="df").item(), matrix)
    elif dist == "gaussian":
        matrix = ot_matrix(dparams.item())
        return ot_map[dist](matrix)
    elif dist == "indep":
        return ot_map[dist](2)
    else:
        return ot_map[dist](dparams.item())


def ot_joint(
    pot_dist: str,
    pot_dparams: xr.DataArray,
    cond_dist: str,
    cond_dparams: xr.DataArray,
    cop_dist: str,
    cop_dparams: xr.DataArray,
) -> ot.ComposedDistribution:
    """Obtain OpenTURNS joint distribution from marginals and copula."""
    marg_pot = ot_marginal(pot_dist, pot_dparams)
    marg_cond = ot_marginal(cond_dist, cond_dparams)
    cop = ot_copula(cop_dist, cop_dparams)
    return ot.ComposedDistribution([marg_pot, marg_cond], cop)


def ot_bic(sample: xr.DataArray, cop_dist: str, cop_dparams: xr.DataArray) -> float:
    """Calculate Bayesian Information Criterion using OpenTURNS."""
    cop = ot_copula(cop_dist, cop_dparams)
    n_param = cop_dparams.notnull().sum().item()
    return ot.FittingTest.BIC(sample.values, cop, n_param)


def ot_aic(sample: xr.DataArray, cop_dist: str, cop_dparams: xr.DataArray) -> float:
    """Calculate Akaike Information Criterion using OpenTURNS."""
    cop = ot_copula(cop_dist, cop_dparams)
    n_param = cop_dparams.notnull().sum().item()
    return ot.FittingTest.AIC(sample.values, cop, n_param)
