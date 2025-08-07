import numpy as np
import xarray as xr

from peach.risk.bootstrap import from_quantile, resample


def test_resample():
    a = xr.DataArray(
        [np.arange(1, 11), np.arange(101, 111)],
        dims=(
            "source",
            "time",
        ),
    )
    out = resample(
        a,
        iteration=3,
        size=6,
        replace=False,
    )
    assert len(out.sample) == 3
    assert len(out.time) == 6
    n = xr.apply_ufunc(
        lambda x: len(np.unique(x)), out, input_core_dims=[["time"]], vectorize=True
    )
    assert (n == 6).all()


def test_from_quantile():
    from scipy.stats import norm

    d = norm()
    x = np.linspace(-3, 3, 100)

    data = xr.DataArray(x, dims=("quantile",), coords={"quantile": d.cdf(x)})

    s = from_quantile(data, 500)
    assert len(s.sample) == 500
    np.testing.assert_almost_equal(s.mean(), 0, 1)
    np.testing.assert_almost_equal(s.var(), 1, 1)
