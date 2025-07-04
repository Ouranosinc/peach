"""Base functions to calculate risk for simple generic hazards."""

import warnings
from functools import partial

import numpy as np
import xarray as xr
import yaml
from scipy import stats

# from mixture import freq_analysis


def extract_infos(yam_file):
    with open(yam_file) as f:
        user_yam = yaml.safe_load(f)
    infos = user_yam["user_inputs"]
    infos["return_periods"] = dict(
        zip(
            list(user_yam["indicators"].keys()),
            [float(x) for x in infos["return_periods"]],
        )
    )
    return infos


def indic_names(ds, infos):  # complete information in user input dictionary
    a = "long_name"
    b = "description"
    if infos["language"] == "french":
        a = a + "_fr"
        b = b + "_fr"
    infos["user_inputs"] = {}
    infos["user_inputs"]["long_name"] = {k: ds[k].attrs[a] for k in ds.keys()}
    infos["user_inputs"]["description"] = {k: ds[k].attrs[b] for k in ds.keys()}
    infos["user_inputs"]["units"] = {k: ds[k].attrs["units"] for k in ds.keys()}
    return infos


def warning_value_0(da) -> None:  # warning if one of the indicator value is equal to 0
    # logcalc = logging.getLogger("risk_calc")
    for v in da.keys():
        va = da[v].where(da[v] == 0)
        j = va.count(dim="time")
        if j > 0:
            warnings.warn(f"Threshold gives value equal to 0 for {v}")


def var_stations(module, yam_dict):
    """Associates variables required to calculate indicators to stations.

    Parameters
    ----------
    yam_dict: dict
        dictionary created from yaml file user inputs
    module: Module type
        xclim module created from yaml file

    Returns
    -------
    Dict associating variables to stations
    """
    dvs = {}
    for key, ind in module.iter_indicators():
        param = list(ind.parameters.keys())
        for p in param:
            if str(ind.parameters[p]["kind"]) == "InputKind.VARIABLE":
                if p == "tasmax" or p == "tasmin" or p == "tas":
                    # ToDo: change index in function of the .csv consutructed file + ADD other variables
                    # is it possible to add distnction for subhourly precip?
                    sta = yam_dict["station"][yam_dict["var"].index("temperature")]
                elif p == "pr":
                    sta = yam_dict["station"][yam_dict["var"].index("precipitation")]
                dvs[p] = sta
    return dvs


def merge_stations_coord(
    ds, da, merge_coords=["station", "station_name", "lat", "lon"], same_coords=["time"]
):
    """Merge stations coordinates / dimensions in ds with da.

    Parameters
    ----------
    ds: xr.Dataset
        xr.Dataset containing stations attributes
    da: xr.DataArray
        xr.DataArray containing indicators
    merge_coords: list
        List of xarray coordinates / dimensions to merge
    same_coords: list
        List of xarray coordinates / dimensions to keep the same

    Returns
    -------
    xr.Dataset
    """
    drop_coords = np.setdiff1d(
        np.array(list(ds.coords.keys())), np.concatenate((merge_coords, same_coords))
    )
    ds = ds.drop_vars(drop_coords)
    for coord in merge_coords:
        ncoord = str(ds[coord].values.item()) + "_" + str(da[coord].values.item())
        ds[coord] = ncoord
    return ds


def _station_preprocess(ds, station):
    ind = np.where(ds.station_name == station)[0].item()
    return ds.isel(station=ind)


def virtual_station(dvs, cat, module):
    """Create xr.Dataset containing each variable required to calculate indicators at each station.

    Parameters
    ----------
    dvs: dict
        dictionary associating variables to stations
    cat: xscen catalog
        Catalog of only one id
    module: Module type
        xclim module created from yaml file

    Returns
    -------
    xr.Dataset

    """
    import xscen as xs

    virt = None
    for var, sta in dvs.items():
        catv = cat.search(variable=var)
        sel_sta = partial(_station_preprocess, station=sta)
        ds_dict = catv.to_dataset_dict(preprocess=sel_sta)
        for k, ds in ds_dict.items():
            da = ds[var]
            if not isinstance(virt, xr.core.dataset.Dataset):
                virt = da.to_dataset()
            else:
                virt = virt.assign({var: da})
                virt = merge_stations_coord(virt, da)

    ind = xs.compute_indicators(virt, module)

    for freq, ds_ind in ind.items():
        add_attrs = {
            k: v
            for k, v in ds.attrs.items()
            if "cat" in k and k not in ind[freq].attrs.keys()
        }
        ind[freq] = xs.clean_up(ds_ind, add_attrs={"global": add_attrs})

    return ind


def _ks(
    data1,
    data2,
    alternative="two-sided",
    method="auto",
    axis=0,
    nan_policy="propagate",
    keepdims=False,
):
    res = stats.ks_2samp(
        data1,
        data2,
        alternative=alternative,
        method=method,
        axis=axis,
        nan_policy=nan_policy,
        keepdims=keepdims,
    )
    return np.array(
        [res.statistic, res.pvalue, res.statistic_location, res.statistic_sign]
    )


def ds_ks(ds_sim, ds_ref, ref_period=None, dim="time", kwargs=None):
    """Returns dataset containing scipy.stats._stats_py.KstestResult for each variable in ds_sim

    Parameters
    ----------
    ds_sim: xr.Dataset
        Indicator simulations
    ds_ref: xr.Dataset
        Indicator observations
    ref_period: list
        start and end date of ref period
    kwargs: dict | None
        kwargs to be sent to scipy.stats.ks_2samp

    Returns
    -------
    xr.Dataset
    """
    if ref_period:
        ds_sim = ds_sim.sel(time=slice(ref_period[0], ref_period[1]))
        ds_ref = ds_ref.sel(time=slice(ref_period[0], ref_period[1]))
        ds_ref["time"] = ds_sim["time"]

    res_ks = xr.apply_ufunc(
        _ks,
        ds_sim,
        ds_ref,
        kwargs=kwargs,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask="parallelized",
        output_core_dims=[["res_ks"]],
        output_dtypes=[float],
        output_sizes={"res_ks": 4},
    )
    res_ks["res_ks"] = ["statistic", "pvalue", "statistic_location", "statistic_sign"]
    return res_ks
