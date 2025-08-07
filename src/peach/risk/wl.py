from __future__ import annotations

import xarray as xr
import xclim as xc
from pyextremes import get_extremes
from xclim.core.indicator import Hourly, Indicator
from xclim.core.locales import load_locale
from xclim.core.units import declare_units
from xclim.indicators.generic._stats import Generic

"""
# Water level indicators

Here are defined indicators for water level computations that serve two purposes:
1. Provide indicator metadata for the frontend (abstract, description, parameters, etc.);
2. Define the computation function that will be used to compute the indicator in the backend.

Note that the actual computation can be `lambda x: x` if we want to return pre-computed data.
"""


class WaterLevels(Hourly):
    """Water level indicator."""

    keywords = "sea level"
    realm = "ocean"


class WLIndicator(Indicator):
    pass


@declare_units(wl="[length]")
def _water_level_max(wl: xr.DataArray, freq: str = "YS") -> xr.DataArray:
    """Compute the maximum water level.

    Resample the original hourly water level data to the specified frequency and compute the maximum value.

    Parameters
    ----------
    wl : xarray.DataArray
        Hourly water level values.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [same units as wl]
        The highest water level value at the given time frequency.
    """
    return xc.indices.generic.select_resample_op(wl, op="max", freq=freq)


@declare_units(wl="[length]")
def _water_level_pot(
    wl: xr.DataArray,
    thresh: float = None,
) -> xr.DataArray:
    """Return water level peaks over threshold.

    Parameters
    ----------
    wl : xarray.DataArray
        Hourly water level values.
    thresh : float
        Threshold to use for computing peaks over threshold.
        Only used if method != None.

    Returns
    -------
    xarray.DataArray, [same units as wl]
        Water level values above a fixed threshold.
    """
    wl_pot_series = get_extremes(ts=wl.to_series(), method="POT", threshold=thresh)
    wl_pot = xr.DataArray(
        data=wl_pot_series,
        coords={"time": wl_pot_series.index},
        dims=["time"],
        name="wl_pot",
        attrs=wl.attrs,
    )
    return wl_pot


@declare_units(wl_pot="[length]")
def _water_level_pot_lambda(
    wl_pot: xr.DataArray,
    thresh: float = None,
) -> xr.DataArray:
    """Return water level peaks over threshold.

    Parameters
    ----------
    wl_pot : xarray.DataArray
        Hourly water level values.

    Returns
    -------
    xarray.DataArray, [same units as wl]
        Water level values above a fixed threshold.
    """
    return wl_pot


wl_max = WaterLevels(
    title="Maximum hourly water level",
    identifier="wl_max",
    units="m",
    standard_name="sea_surface_height_above_geopotential_datum",
    long_name="Maximum hourly water level",
    description="{freq} maximum water level",
    abstract="Maximum hourly water level",
    cell_methods="time: maximum",
    compute=_water_level_max,
)


wl_pot = Generic(
    realm="ocean",
    title="Peaks over threshold",
    identifier="wl_pot",
    units="m",
    standard_name="sea_surface_height_above_geopotential_datum",
    long_name="Water level peaks over threshold",
    description="Water level peaks over threshold",
    abstract="Water level peaks over threshold",
    compute=_water_level_pot_lambda,
)

translation = {
    "peach.risk.wl.WL_MAX": {
        "long_name": "Niveau d'eau horaire maximal",
        "description": "Niveau d'eau maximal horaire",
        "abstract": "Niveau d'eau horaire maximal",
        "title": "Niveau d'eau horaire maximal",
    },
    "WL_POT": {
        "long_name": "Pics de niveau d'eau au-dessus du seuil",
        "description": "Pics de niveau d'eau au-dessus du seuil",
        "abstract": "Pics de niveau d'eau au-dessus du seuil",
        "title": "Niveau d'eau au-dessus du seuil",
    },
}


load_locale(translation, "fr")
