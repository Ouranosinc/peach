import numpy as np
import scipy.stats
import lmo
import xarray as xr
from xclim.core.formatting import update_history
from scipy.stats import rv_continuous


def fit(da, dist: str | rv_continuous="norm", method: str="PWM", dim="time"):
    """Fit a distribution to the data array along the specified dimension.

    Parameters
    ----------
    da : xarray.DataArray
      The data array to fit.
    dist : str or scipy.stats.rv_continuous, optional
      The distribution to fit. Can be a string name of a scipy distribution or a scipy.stats distribution object.
      Default is "norm" (normal distribution).
    method : str, optional
      The fitting method to use. Can be "PWM" (probability weighted moments) or "ML" (maximum likelihood).
    dim : str, optional
      The dimension along which to fit the distribution. Default is "time".

    Returns
    -------
    xarray.DataArray
      An array of fitted distribution parameters with a new dimension "dparams" for the parameters.
      The array will have an attribute "scipy_dist" storing the name of the fitted distribution.
    """
    method = method.upper()
    method_name = {
        "ML": "maximum likelihood",
        "MM": "method of moments",
        "MLE": "maximum likelihood",
        "MSE": "maximum product of spacings",
        "MPS": "maximum product of spacings",
        "PWM": "probability weighted moments",
        "APP": "approximative method",
    }

    if isinstance(dist, str):
        dc = getattr(scipy.stats, dist, None)
        if dc is None:
            raise ValueError(f"Distribution {dist} not found in scipy.stats.")
        
    elif isinstance(dist, scipy.stats.rv_continuous):
        dc = dist
        dist = dist.name
    else:
        raise ValueError("dist must be a string or a scipy.stats.rv_continuous object.")

    shape_params = [] if dc.shapes is None else dc.shapes.split(",")
    dist_params = shape_params + ["loc", "scale"]
    nparams = len(dist_params)

    def fit_func(x):
        x = np.ma.masked_invalid(x).compressed()  # pylint: disable=no-member

        # Return NaNs if array is empty.
        if len(x) <= 1:
            return np.asarray([np.nan] * nparams)

        if method == "PWM":
            out = dc.l_fit(x)
        else:
            out = dc.fit(x, method=method)
        
        params = np.asarray(out)
        # Fill with NaNs if one of the parameters is NaN
        if np.isnan(params).any():
            params[:] = np.nan
        return params


    params = xr.apply_ufunc(
        fit_func,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["dparams"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
        dask_gufunc_kwargs={"output_sizes": {"dparams": len(dist_params)}},
    )

    # Add coordinates for the distribution parameters and transpose to original shape (with dim -> dparams)
    dims = [d if d != dim else "dparams" for d in da.dims]
    out = params.assign_coords(dparams=dist_params).transpose(*dims)
    attrs = {
        "long_name": f"{dc.name} parameters",
        "description": f"Parameters of the {dc.name} distribution",
        "method": method,
        "estimator": method_name[method].capitalize(),
        "scipy_dist": dc.name,
        "units": "",
        "history": update_history(
            f"Estimate distribution parameters by {method_name[method]} method along dimension {dim}.",
            new_name="fit",
            data=da,
        ),
    }
    out.attrs.update(attrs)
    return out

