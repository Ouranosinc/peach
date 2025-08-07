from __future__ import annotations

import numpy as np
import xarray as xr

"""Utilities to help with bootstraping.

Some code inspired from xskillscore.
"""


def _gen_idx(
    da: xr.DataArray,
    iteration: int,
    size: int,
    replace: bool,
) -> xr.DataArray:
    """Generate indices to select from.

    Parameters
    ----------
    da : xr.DataArray
        DataArray coordinate to generate indices for, e.g. time.
    iteration : int
        Number of samples to draw.
    size : int
        Number of items in one sample.
    replace : bool
        Whether to sample with replacement.
    """
    size = size or len(da)

    if replace:
        idx = np.random.randint(0, da.size, (iteration, size))
    else:
        rng = np.random.Generator(np.random.PCG64())
        idx = np.array(
            [rng.choice(da.size, size, replace=replace) for i in range(iteration)]
        )

    return xr.DataArray(
        idx,
        dims=("sample", da.name),
        coords=({"sample": range(iteration)}),
    )


def resample(da, iteration, size=None, replace=True, dim="time"):
    """Resample a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to resample.
    iteration : int
        Number of samples to draw.
    size : int
        Number of items in one sample. Defaults to the length of the dimension.
    replace : bool
        Whether to sample with replacement. Defaults to True.
    """
    size = size or len(da[dim])

    if size == len(da[dim]) and replace is False and iteration > 1:
        raise ValueError(
            "It's not really useful to sample without replacement more than once if `size` is the same."
        )

    idx = _gen_idx(da[dim], iteration, size, replace)
    return da.isel({dim: idx})


def from_quantile(cdf, iteration: int = 1):
    """
    Return a random sample given a CDF.

    Parameters
    ----------
    cdf : xr.DataArray
        Cumulative distribution function with a `quantile` coordinate.
    iteration : int
        Number of samples to draw.

    Returns
    -------
    xr.DataArray
        Random samples from the distribution.

    Notes
    -----
    Based on Gabriel xhydro
            https://xhydro.readthedocs.io/en/latest/notebooks/climate_change.html#Use-Case-#2:-Probabilistic-reference-data
    """
    rng = np.random.Generator(np.random.PCG64())

    # Calculate the weights for each percentile in the distribution
    q = cdf["quantile"].values

    # Ideally, logic for endpoints that are not 0 or 1 should in included, but it's not needed in this application
    # One approach is to add 0 and 1 to the quantiles (below), as long as values are added to the cdf as well (not shown).
    # if q[0] != 0:
    #     q = np.concatenate(([0], q))
    # if q[-1] != 1:
    #     q = np.concatenate((q, [1]))

    w = np.diff(q) / 2
    # Double the first and last weights
    first = w[0] * 2
    last = w[-1] * 2
    w = w[:-1] + w[1:]
    w = np.concatenate(([first], w, [last]))
    w /= w.sum()

    # Apply the sampling function across the realization dimension
    out = xr.apply_ufunc(
        rng.choice,
        cdf,
        input_core_dims=[["quantile"]],
        output_core_dims=[["sample"]],
        vectorize=True,
        kwargs=dict(size=iteration, p=w, replace=True),
    )

    out = out.assign_coords(sample=np.arange(iteration))
    out.attrs = cdf.attrs
    return out
