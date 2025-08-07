from __future__ import annotations

import xarray as xr
from xclim.core.locales import load_locale
from xclim.core.units import declare_units
from xclim.indicators.generic._stats import Generic

__all__ = ["idf"]


"""
# IDF indicator

This is a bit of a hack. We used this lambda x: x indicator here to select the duration of the IDF data,
and feed some attributes to the backend.
"""


@declare_units(idf="[precipitation]", tas="[temperature]")
def _lambda(
    idf: xr.DataArray, tas: xr.DataArray = None, duration: str = "24h"
) -> xr.DataArray:
    """Maximum annual precipitation for the given duration.

    Return the maximum annual precipitation aggregated over a given duration.

    Parameters
    ----------
    idf: xarray.DataArray
        Maximum annual precipitation.
    duration : str
        Duration of rainfall.

    Returns
    -------
    xarray.DataArray
        Maximum annual precipitation for the given duration.
    """
    return idf.sel(duration=duration)


idf = Generic(
    realm="atmos",
    title="Extreme rainfall",
    identifier="idf",
    units="m/h",
    standard_name="thickness_of_rainfall_amount",
    long_name="Maximum rainfall depth",
    description="Maximum rainfall depth over {duration}",
    abstract="Maximum annual rainfall",
    compute=_lambda,
)

translation = {
    "IDF": {
        "long_name": "Pluie maximale annuelle",
        "description": "Pluie maximale annuelle de durée {duration}",
        "abstract": "Pluie maximale annuelle",
        "title": "Pluie maximale annuelle",
    }
}


load_locale(translation, "fr")
