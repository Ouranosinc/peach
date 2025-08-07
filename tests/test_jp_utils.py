"""Missing tests TODO for completion:
- ot_copula, ot_joint (see copula.py)
- wl_norm
"""

import numpy as np
import openturns as ot
import pytest
import scipy
import xarray as xr

pytest.importorskip("copulae")

from peach.frontend.cbcl_utils import define_q, matching_events  # noqa: E402
from peach.frontend.copula import ot_marginal  # noqa: E402


def test_define_q():
    parent_array = np.array([10, 20, 30, 40, 50])
    child_array = np.array([10, 20, 30, 40, 50])
    expected_q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = define_q(parent_array, child_array)
    assert np.allclose(result, expected_q)

    # Not sorted
    parent_array = np.array([30, 10, 50, 20, 40])
    child_array = np.array([10, 20, 30, 40, 50])
    expected_q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = define_q(parent_array, child_array)
    assert np.allclose(result, expected_q)

    # Outside limits
    parent_array = np.array([10, 20, 30, 40, 50])
    child_array = np.array([10, 20, 30, 40, 60])
    expected_q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    with pytest.raises(
        ValueError,
        match="All values in child_array must be within the range of parent_array.",
    ):
        define_q(parent_array, child_array)

    # Interpolated
    parent_array = np.array([10, 20, 30, 40, 50])
    child_array = np.array([15, 20, 30, 40, 50])
    expected_q = np.array([0.125, 0.25, 0.5, 0.75, 1.0])
    result = define_q(parent_array, child_array)
    assert np.allclose(result, expected_q)


def test_get_matching_events(synthetic_ewl_ds, synthetic_ds_daily):
    _, wl_pot, _, _ = synthetic_ewl_ds
    pr_sim = synthetic_ds_daily

    pr_timeseries = pr_sim.sel(time=slice("1995-01-01", "2014-12-31")).isel(
        realization=0
    )
    pot_da_clean, pr_cond = matching_events(wl_pot, pr_timeseries)

    assert (pr_cond.time.values >= pr_timeseries.time.values.min() - np.timedelta64(1, 'D')).all()
    assert (pr_cond.time.values <= pr_timeseries.time.values.max() + np.timedelta64(1, 'D')).all()
    assert (pr_cond.values >= pr_timeseries.values.min()).all()
    assert (pr_cond.values <= pr_timeseries.values.max()).all()
    assert len(pr_cond) == len(pot_da_clean)


### MARGINAL (UNIVARIATE) DISTRIBUTIONS
def create_xarray_dparams(params):
    """Create dprams xarray for testing."""
    data = np.array([params[key] for key in params.keys()])
    coords = {"dparams": list(params.keys())}
    return xr.DataArray(data, coords=coords, dims=["dparams"])


def test_normal_mapping():
    dparams = create_xarray_dparams({"loc": 0.0, "scale": 1.0})
    ot_dist = ot_marginal("norm", dparams)

    assert isinstance(ot_dist, ot.Normal)
    params = ot_dist.getParameter()
    assert params[0] == dparams.sel(dparams="loc").item()
    assert params[1] == dparams.sel(dparams="scale").item()

    test_value = 1.5
    scipy_norm = scipy.stats.norm(loc=dparams.values[0], scale=dparams.values[1])
    assert np.isclose(ot_dist.computePDF(test_value), scipy_norm.pdf(test_value))
    assert np.isclose(ot_dist.computeCDF(test_value), scipy_norm.cdf(test_value))


def test_student_mapping():
    dparams = create_xarray_dparams({"df": 10, "loc": 0.0, "scale": 1.0})
    ot_dist = ot_marginal("t", dparams)

    assert isinstance(ot_dist, ot.Student)
    params = ot_dist.getParameter()
    assert params[0] == dparams.sel(dparams="df").item()
    assert params[1] == dparams.sel(dparams="loc").item()
    assert params[2] == dparams.sel(dparams="scale").item()

    test_value = 1.5
    scipy_t = scipy.stats.t(
        df=dparams.values[0], loc=dparams.values[1], scale=dparams.values[2]
    )
    assert np.isclose(ot_dist.computePDF(test_value), scipy_t.pdf(test_value))
    assert np.isclose(ot_dist.computeCDF(test_value), scipy_t.cdf(test_value))


def test_gamma_mapping():
    dparams = create_xarray_dparams(
        {"a": 2.0, "loc": 1.0, "scale": 2.0}
    )  # scipy: shape, loc, scale
    ot_dist = ot_marginal("gamma", dparams)

    assert isinstance(ot_dist, ot.Gamma)
    params = ot_dist.getParameter()
    assert params[0] == dparams.sel(dparams="a").item()  # ot: shape
    assert params[1] == 1 / dparams.sel(dparams="scale").item()  # ot: rate
    assert params[2] == dparams.sel(dparams="loc").item()  # ot: shift

    test_value = 1.5
    scipy_gamma = scipy.stats.gamma(
        a=dparams.values[0], loc=dparams.values[1], scale=dparams.values[2]
    )
    assert np.isclose(ot_dist.computePDF(test_value), scipy_gamma.pdf(test_value))
    assert np.isclose(ot_dist.computeCDF(test_value), scipy_gamma.cdf(test_value))


def test_genextreme_mapping():
    # scipy: c (shape), loc (location), scale (scale)
    # ot: mu (location), sigma (scale), xi (shape)
    dparams = create_xarray_dparams({"c": 0.1, "loc": 1.0, "scale": 2.0})
    ot_dist = ot_marginal("genextreme", dparams)

    assert isinstance(ot_dist, ot.GeneralizedExtremeValue)
    ot_params = ot_dist.getParameter()
    assert ot_params[0] == dparams.sel(dparams="loc").item()
    assert ot_params[1] == dparams.sel(dparams="scale").item()
    assert ot_params[2] == -dparams.sel(dparams="c").item()  # shape

    test_value = 1.5
    scipy_genextreme = scipy.stats.genextreme(
        c=dparams.values[0], loc=dparams.values[1], scale=dparams.values[2]
    )
    assert np.isclose(ot_dist.computePDF(test_value), scipy_genextreme.pdf(test_value))
    assert np.isclose(ot_dist.computeCDF(test_value), scipy_genextreme.cdf(test_value))


def test_genpareto_mapping():
    dparams = create_xarray_dparams({"c": 0.1, "loc": 1.0, "scale": 2.0})
    ot_dist = ot_marginal("genpareto", dparams)

    assert isinstance(ot_dist, ot.GeneralizedPareto)
    ot_params = ot_dist.getParameter()
    assert ot_params[0] == dparams.sel(dparams="scale").item()
    assert ot_params[1] == dparams.sel(dparams="c").item()
    assert ot_params[2] == dparams.sel(dparams="loc").item()

    test_value = 1.5
    scipy_dist = scipy.stats.genpareto(
        c=dparams.values[0], loc=dparams.values[1], scale=dparams.values[2]
    )
    assert np.isclose(ot_dist.computePDF(test_value), scipy_dist.pdf(test_value))
    assert np.isclose(ot_dist.computeCDF(test_value), scipy_dist.cdf(test_value))


def test_lognorm_mapping():
    dparams = create_xarray_dparams({"s": 0.954, "loc": 1.0, "scale": np.exp(2.0)})
    ot_dist = ot_marginal("lognorm", dparams)

    assert isinstance(ot_dist, ot.LogNormal)
    ot_params = ot_dist.getParameter()
    assert ot_params[0] == np.log(
        dparams.sel(dparams="scale").item()
    )  # np.log(params[2])
    assert ot_params[1] == dparams.sel(dparams="s").item()  # params[0]
    assert ot_params[2] == dparams.sel(dparams="loc").item()  # params[1]

    test_value = 1.5
    scipy_lognorm = scipy.stats.lognorm(
        s=dparams.values[0], loc=dparams.values[1], scale=dparams.values[2]
    )
    assert np.isclose(ot_dist.computePDF(test_value), scipy_lognorm.pdf(test_value))
    assert np.isclose(ot_dist.computeCDF(test_value), scipy_lognorm.cdf(test_value))


def test_uniform_mapping():
    dparams = create_xarray_dparams({"loc": 5.0, "scale": 10.0})
    ot_dist = ot_marginal("uniform", dparams)

    assert isinstance(ot_dist, ot.Uniform)
    ot_params = ot_dist.getParameter()
    assert ot_params[0] == dparams.sel(dparams="loc").item()
    assert (
        ot_params[1]
        == dparams.sel(dparams="loc").item() + dparams.sel(dparams="scale").item()
    )

    test_value = 7.5
    scipy_uniform = scipy.stats.uniform(
        loc=dparams.sel(dparams="loc").item(), scale=dparams.sel(dparams="scale").item()
    )
    assert np.isclose(ot_dist.computePDF(test_value), scipy_uniform.pdf(test_value))
    assert np.isclose(ot_dist.computeCDF(test_value), scipy_uniform.cdf(test_value))
