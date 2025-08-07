import numpy as np
import xarray as xr
import xclim as xc
import xclim.indices.stats
from xclim.core.formatting import unprefix_attrs, update_history


def parametric_pdf(p, v):
    """Return the cumulative distribution function corresponding to the given distribution parameters and value.

    Parameters
    ----------
    p : xr.DataArray
      Distribution parameters returned by the `fit` function.
      The array should have dimension `dparams` storing the distribution parameters,
      and attribute `scipy_dist`, storing the name of the distribution.
    v : Union[float, Sequence]
      Value to compute the PDF.

    Returns
    -------
    xarray.DataArray
      An array of parametric PDF values estimated from the distribution parameters.
    """
    v = np.atleast_1d(v)

    # Get the distribution
    dist = p.attrs["scipy_dist"]
    dc = xc.indices.stats.get_dist(dist)

    # Create a lambda function to facilitate passing arguments to dask. There is probably a better way to do this.
    def func(x):
        return dc.pdf(v, *x)

    data = xr.apply_ufunc(
        func,
        p,
        input_core_dims=[["dparams"]],
        output_core_dims=[["pdf"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
        dask_gufunc_kwargs={"output_sizes": {"pdf": len(v)}},
    )

    # Assign quantile coordinates and transpose to preserve original dimension order
    dims = [d if d != "dparams" else "pdf" for d in p.dims]
    out = data.assign_coords(pdf=v).transpose(*dims)
    out.attrs = unprefix_attrs(p.attrs, ["units", "standard_name"], "original_")

    attrs = dict(
        long_name=f"{dist} pdf",
        description=f"PDF estimated by the {dist} distribution",
        cell_methods="dparams: pdf",
        history=update_history(
            "Compute parametric pdf from distribution parameters",
            new_name="parametric_pdf",
            parameters=p,
        ),
    )
    out.attrs.update(attrs)
    return out


def parametric_logpdf(p, v):
    """Return the cumulative distribution function corresponding to the given distribution parameters and value.

    Parameters
    ----------
    p : xr.DataArray
      Distribution parameters returned by the `fit` function.
      The array should have dimension `dparams` storing the distribution parameters,
      and attribute `scipy_dist`, storing the name of the distribution.
    v : Union[float, Sequence]
      Value to compute the PDF.

    Returns
    -------
    xarray.DataArray
      An array of parametric PDF values estimated from the distribution parameters.
    """
    v = np.atleast_1d(v)

    # Get the distribution
    dist = p.attrs["scipy_dist"]
    dc = xc.indices.stats.get_dist(dist)

    # Create a lambda function to facilitate passing arguments to dask. There is probably a better way to do this.
    def func(x):
        return dc.logpdf(v, *x)

    data = xr.apply_ufunc(
        func,
        p,
        input_core_dims=[["dparams"]],
        output_core_dims=[["logpdf"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
        dask_gufunc_kwargs={"output_sizes": {"logpdf": len(v)}},
    )

    # Assign quantile coordinates and transpose to preserve original dimension order
    dims = [d if d != "dparams" else "logpdf" for d in p.dims]
    out = data.assign_coords(logpdf=v).transpose(*dims)
    out.attrs = unprefix_attrs(p.attrs, ["units", "standard_name"], "original_")

    attrs = dict(
        long_name=f"{dist} logpdf",
        description=f"Log PDF estimated by the {dist} distribution",
        cell_methods="dparams: logpdf",
        history=update_history(
            "Compute parametric logpdf from distribution parameters",
            new_name="parametric_logpdf",
            parameters=p,
        ),
    )
    out.attrs.update(attrs)
    return out


def values_cdf(p, qmin=0.00001, qmax=0.99999, pts=10000):  # returns CDF values
    mmin = xc.indices.stats.parametric_quantile(p, qmin).min(dim="sims")
    mmax = xc.indices.stats.parametric_quantile(p, qmax).max(dim="sims")
    return np.squeeze(np.linspace(mmin, mmax, pts))
