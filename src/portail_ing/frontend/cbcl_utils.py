import numpy as np
import xarray as xr


def define_q(parent_array: np.array, child_array: np.array) -> np.array:
    """Computes the quantiles for each value in `child_array` based on its position within `parent_array`."""
    if child_array.min() < parent_array.min() or child_array.max() > parent_array.max():
        raise ValueError(
            "All values in child_array must be within the range of parent_array."
        )
    sorted_parent = np.sort(parent_array)
    q = np.interp(child_array, sorted_parent, np.linspace(0, 1, len(parent_array)))
    return q


def wl_norm(sl_mm_yr: float, ref_period: tuple):
    """Get the mean water level difference between AR6 and the specified reference period.

    Difference in mean water level is obtained from the historical rate of sea-level change.

    If a station has a positive sea-level change rate (sea-level rise), the water level
    difference (wl_diff_m) will be negative if the reference period provided (e.g., 1961-1990)
    is older than that of of the AR6 reference period, and it will be positive if the reference
    period provided is more recent than the AR6 reference period.
    """
    ar6_period = [1995, 2014]

    ar6_middle_yr = (ar6_period[0] + ar6_period[1]) / 2
    ref_middle_yr = (ref_period[0] + ref_period[1]) / 2

    wl_diff_m = (ref_middle_yr - ar6_middle_yr) * sl_mm_yr / 1000

    return wl_diff_m


def matching_events(pot_da: xr.DataArray, timeseries_da: xr.DataArray) -> tuple:
    """Identifies matching extreme events in `timeseries_da` around the timepoints in `pot_da`."""
    LAG_DAYS = 1
    max_values, max_times = [], []
    for time in pot_da["time"].values:
        start_time = np.datetime64(time) - np.timedelta64(LAG_DAYS, "D")
        end_time = np.datetime64(time) + np.timedelta64(LAG_DAYS, "D")
        window_values = timeseries_da.sel(time=slice(start_time, end_time))

        if len(window_values) > 0:
            max_value = window_values.max().values
            max_time = time
        else:
            max_value = np.nan
            max_time = time

        max_values.append(max_value)
        max_times.append(max_time)

    da_cond = xr.DataArray(
        max_values,
        coords={"time": max_times},
        dims="time",
        name=(
            "pr_cond"
            if timeseries_da.name in {"pr_sim_daily", "pr"}
            else f"{timeseries_da.name}_cond"
        ),
        attrs=timeseries_da.attrs,
    )

    da_cond.attrs["ex_type"] = "Conditional extremes"
    
    if timeseries_da.name in {"pr_sim_daily", "pr"}:
        # Remove events where precipitation < 1 mm
        mask = ~np.isnan(pot_da.values) & ~np.isnan(da_cond.values) & (da_cond.values >= 1)
    else:
        mask = ~np.isnan(pot_da.values) & ~np.isnan(da_cond.values)
    pot_da_cleaned = pot_da.sel(time=pot_da.time.values[mask])
    da_cond_cleaned = da_cond.sel(time=da_cond.time.values[mask])

    return pot_da_cleaned, da_cond_cleaned
