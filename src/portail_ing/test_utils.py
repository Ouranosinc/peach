import io
import itertools
import threading
import zipfile

import numpy as np
import openturns as ot
import pandas as pd
import xarray as xr
import zarr
from pyextremes import get_extremes


def synthetic_ds():

    ds = xr.Dataset()
    time = xr.cftime_range(start="1950-01-01", end="2020-12-31", freq="YE")

    for i, uuid in enumerate(["00", "11"]):
        ds[uuid] = xr.DataArray(
            data=np.random.rand(len(time)),
            coords={"time": time},
            dims=["time"],
            attrs={
                "long_name": f"Var en {i}",
                "long_name_fr": f"Var fr {i}",
                "description": f"Description of indicator {i}",
                "description_fr": f"Description de l'indicateur {i}",
                "stations": {"pr": "7028441"},
                "id": "XIND",
                "params": {"thresh": "1.0 mm/d"},
                "history": f"[2024-11-18 20:37:17] {uuid}: XIND(pr=pr, thresh='1.0 mm/d', freq='YS') with options check_missing=skip - xclim version: 0.53.2",
            },
        )
    return ds.data_vars


def synthetic_ds_fut():
    ds = xr.Dataset()
    time = xr.cftime_range(start="1900-01-01", end="2100-12-31", freq="YE")
    source_id = ["INM-CM4-8", "GFDL-CM4", "HadGEM3-GC31-MM", "MIROC6", "TaiESM1"]
    experiment_id = ["ssp126", "ssp245", "ssp370", "ssp585"]
    variant_label = ["r1i1p1f1", "r2i1p1f1"]

    for i, uuid in enumerate(["00", "11"]):
        data = np.random.rand(
            len(time),
            len(source_id),
            len(experiment_id),
            len(variant_label),
        )
        data[:, 0, 0, 0] = np.nan
        data[-80:] += 0.1

        da = xr.DataArray(
            data=data,
            coords={
                "time": time,
                "source_id": source_id,
                "experiment_id": experiment_id,
                "variant_label": variant_label,
            },
            dims=["time", "source_id", "experiment_id", "variant_label"],
            attrs={
                "long_name": f"Var en {i}",
                "long_name_fr": f"Var fr {i}",
                "description": f"Description of indicator {i}",
                "description_fr": f"Description de l'indicateur {i}",
                "stations": {"pr": "7028441"},
                "id": "XIND",
                "params": {"thresh": "1.0 mm/d"},
                "history": f"[2024-11-18 20:37:17] {uuid}: XIND(pr=pr, thresh='1.0 mm/d', freq='YS') with options check_missing=skip - xclim version: 0.53.2",
            },
        )
        ds[uuid] = da.stack(realization=("variant_label", "source_id", "experiment_id"))
    out = ds.data_vars
    return out


def synthetic_ds_daily():
    time_range = pd.date_range("1960-01-01", "2100-12-31", freq="D")
    variant_labels = ["r1i1p1f1"]
    source_ids = ["INM-CM4-8", "HadGEM3-GC31-MM"]
    experiment_ids = ["ssp245", "ssp370", "ssp585"]

    realization_tuples = list(
        itertools.product(variant_labels, source_ids, experiment_ids)
    )
    # Add a realization for one but not the others
    realization_tuples.append(("r1i1p1f1", "HadGEM3-GC31-MM", "ssp126"))
    realization_index = pd.MultiIndex.from_tuples(
        realization_tuples, names=["variant_label", "source_id", "experiment_id"]
    )

    max_values = 200 + (100 * (np.arange(len(time_range)) / len(time_range)))
    random_data = np.random.rand(len(realization_index), len(time_range)) * max_values

    pr_sim = xr.DataArray(
        random_data,
        dims=["realization", "time"],
        coords={
            "realization": realization_index,
            "time": time_range,
        },
        name="pr_sim_daily",
        attrs={
            "bias_adjustment": "Dummy_Adjustment",
            "cell_measures": "area: areacella",
            "cell_methods": "time: mean within days",
            "comment": "includes both liquid and solid phases",
            "description": "Daily precipitation flux at surface",
            "description_fr": "Précipitation quotidienne à la surface",
            "interval_operation": "900 s",
            "interval_write": "1 d",
            "long_name": "Mean daily precipitation flux",
            "long_name_fr": "Précipitation journalière moyenne",
            "online_operation": "average",
            "original_name": "mo: (dummy_stash, lbproc: 128)",
            "standard_name": "precipitation_flux",
            "units": "mm day-1",
            "stations": {"pr": "7028441"},
            "id": "XIND",
            "params": {"thresh": "1.0 mm/d"},
            "history": "[2024-11-18 20:37:17] pr_sim_daily: XIND(pr=pr, thresh='1.0 mm/d', freq='YS') with options check_missing=skip - xclim version: 0.53.2",
        },
    )

    return pr_sim


def synthetic_ewl_ds():
    time_wl = pd.date_range(start="1960-01-01", end="2012-12-31", freq="h")
    wl_data = np.random.randn(len(time_wl))
    stn_thresh = np.quantile(wl_data, 0.98)
    attrs = {
        "wl_stn_name": "NAME",
        "wl_stn_id": "00490",
        "label": "Dummy label",
        "sl_change": 0,
        "stn_thresh": stn_thresh,
        "sl_mm_yr": 2,
        "peaks_per_yr": 365 * 0.02,
        "ar6ref_wl": -0.2,
        "ref_period": [1995, 2014],
        "units": "m",
        "standard_name": "sea_surface_height_above_geopotential_datum",
        "long_name": "Sea surface height above geopotential datum",
        "long_name_fr": "Niveau de la mer au-dessus du géopotentiel",
        "description": "Dummy description",
        "description_fr": "Description bidon",
        "geopotential_datum_name": "Canadian Geodetic Vertical Datum of 2013 (CGVD2013)",
        "stn_nyrs": 10,
        "ex_type": "Peaks over threshold",
        "stations": {"wl": "00490"},
        "id": "XIND",
        "params": {"thresh": "1.0 mm/d"},
        "history": "[2024-11-18 20:37:17] wl_pot: XIND(pr=pr, thresh='1.0 mm/d', freq='YS') with options check_missing=skip - xclim version: 0.53.2",
    }
    wl = xr.DataArray(
        data=wl_data, coords={"time": time_wl}, dims=["time"], name="wl", attrs=attrs
    )
    wl_pot_series = get_extremes(ts=wl.to_series(), method="POT", threshold=stn_thresh)
    wl_pot = xr.DataArray(
        data=wl_pot_series,
        coords={"time": wl_pot_series.index},
        dims=["time"],
        name="wl_pot",
        attrs=attrs,
    )

    with xr.set_options(keep_attrs=True):
        wl = wl.dropna(dim="time")
        wl = wl.rename("wl")
        # wl = wl.rename({"time": "time_ref"})

    with xr.set_options(keep_attrs=True):
        wl_pot = wl_pot.dropna(dim="time")
        wl_pot = wl_pot.rename("wl_pot")
        # wl_pot = wl_pot.rename({"time": "time_ref"})
        # wl_pot = wl_pot.expand_dims({"realization": ["obs"], "period": ["ref"]})

    time_daily = pd.date_range(start="2020-01-01", end="2150-01-01", freq="D")
    base_q = np.arange(0, 1.05, 0.05)
    extra_q = np.array(
        [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
    )
    quantiles = np.sort(np.concatenate((base_q, extra_q)))

    experiment_id = [
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp585",
    ]

    slopes = [2, 3, 4, 5]
    orig_data = np.array(
        [
            slope * (time_daily.year - 2020) / 1000
            + np.random.randn(len(time_daily)) * 0.001
            for slope in slopes
        ]
    )

    da_daily = xr.DataArray(
        orig_data,
        coords=[("experiment_id", experiment_id), ("time", time_daily)],
        name="daily_data",
    )

    def quantile_calculation(da_decadal, da_daily, quantiles):
        quantile_results = np.full(
            (len(da_decadal.experiment_id), len(da_decadal.time), len(quantiles)), np.nan
        )
        for i, real in enumerate(da_decadal.experiment_id):
            for j, time in enumerate(da_decadal.time):
                decade_start = time.values
                decade_end = decade_start + pd.DateOffset(years=10)
                data_slice = da_daily.sel(
                    experiment_id=real, time=slice(decade_start, decade_end)
                )
                quantile_values = np.quantile(data_slice, quantiles, axis=0)
                quantile_results[i, j, :] = quantile_values
        return quantile_results

    da_decadal = da_daily.resample(time="10YS").mean()
    sl_delta_data = quantile_calculation(da_decadal, da_daily, quantiles)
    time_decadal = pd.date_range(start="2020-01-01", end="2150-01-01", freq="10YS")

    sl_attrs = {
        "long_name": "Sea-level delta",
        "long_name_fr": "Delta du niveau de la mer",
        "description": "Dummy description",
        "description_fr": "Description bidon",
        "standard_name": "sea_surface_height_above_geoid",
        "units": "m",
        "grid_mapping": "spatial_ref",
        "sl_mm_yr": 2.8,
        "ref_period": [1995, 2014],
        "stations": {"wl": "00490"},
        "id": "XIND",
        "params": {"thresh": "1.0 mm/d"},
        "history": "[2024-11-18 20:37:17] sl_delta: XIND(pr=pr, thresh='1.0 mm/d', freq='YS') with options check_missing=skip - xclim version: 0.53.2",
    }

    sl = xr.DataArray(
        data=sl_delta_data,
        coords={
            "time": time_decadal,
            "quantile": quantiles,
            "experiment_id": experiment_id,
        },
        dims=["experiment_id", "time", "quantile"],
        name="sl_delta",
        attrs=sl_attrs,
    )

    return wl, wl_pot, sl, stn_thresh


def synthetic_jp_ds(ind="wl_prcond"):
    joint_dist = ot.ComposedDistribution(
        [ot.GeneralizedPareto(1.0, 0.0, 5.0), ot.Normal(1.0, 1.0)],
        ot.ClaytonCopula(2),
    )
    sample = joint_dist.getSample(500)
    pot_data = [x[0] for x in sample]
    cond_data = [x[1] for x in sample]
    times = xr.cftime_range(start="1960-01-01", end="2011-12-31", freq="D")

    time = np.sort(np.random.choice(times, size=500, replace=False))
    wl_attrs = {
        "wl_stn_name": "HALIFAX",
        "wl_stn_id": "00490",
        "ar6ref_wl": -0.2,
        "ref_period": [1995, 2014],
        "units": "m",
        "standard_name": "sea_surface_height_above_geopotential_datum",
        "geopotential_datum_name": "Canadian Geodetic Vertical Datum of 2013 (CGVD2013)",
        "stn_nyrs": 51,
        "stations": {"wl": "00490"},
        "id": "XIND",
        "params": {"thresh": "1.0 mm/d"},
        "history": "[2024-11-18 20:37:17] wl_pot: XIND(pr=pr, thresh='1.0 mm/d', freq='YS') with options check_missing=skip - xclim version: 0.53.2",
    }
    pr_attrs = {
        "units": "mm day-1",
        "standard_name": "precipitation_flux",
        "comment": "ECCC Second Generation of Homogenized Precipitation Data",
        "ahccd_id": 8403506,
        "ahccd_name": "ST JOHN'S",
        "label": "Precipitation Extremes Conditional on Water Levels",
        "stations": {"pr": "7028441"},
        "id": "XIND",
        "params": {"thresh": "1.0 mm/d"},
        "history": "[2024-11-18 20:37:17] pr_cond: XIND(pr=pr, thresh='1.0 mm/d', freq='YS') with options check_missing=skip - xclim version: 0.53.2",
    }
    obs_pot = xr.DataArray(
        data=pot_data,
        coords={"time": time},
        dims=["time"],
    )
    obs_cond = xr.DataArray(
        data=cond_data,
        coords={"time": time},
        dims=["time"],
    )
    if ind == "wl_prcond":
        obs_pot.attrs = wl_attrs
        obs_pot.name = "wl_pot"
        obs_cond.attrs = pr_attrs
        obs_cond.name = "pr_cond"
        with xr.set_options(keep_attrs=True):
            obs_pot.attrs["sl_mm_yr"] = 2
            obs_pot.attrs["long_name"] = "Water level peaks over threshold"
            obs_pot.attrs["long_name_fr"] = "Pics de niveaux d'eau au-dessus du seuil"
            obs_pot.attrs["description"] = "Dummy description"
            obs_pot.attrs["description_fr"] = "Description bidon"
            obs_cond.attrs["long_name"] = (
                "Precipitation extremes conditional on water level extremes"
            )
            obs_cond.attrs["long_name_fr"] = (
                "Précipitations extrêmes conditionnelles aux extrêmes de niveaux d'eau"
            )
            obs_cond.attrs["description"] = "Dummy description"
            obs_cond.attrs["description_fr"] = "Description bidon"
    elif ind == "pr_wlcond":
        obs_pot.attrs = pr_attrs
        obs_pot.name = "pr_pot"
        obs_cond.attrs = wl_attrs
        obs_cond.name = "wl_cond"
        with xr.set_options(keep_attrs=True):
            obs_cond.attrs["sl_mm_yr"] = 2
            obs_pot.attrs["long_name"] = "Precipitation peaks over threshold"
            obs_pot.attrs["long_name_fr"] = "Pics de précipitations au-dessus du seuil"
            obs_pot.attrs["description"] = "Dummy description"
            obs_pot.attrs["description_fr"] = "Description bidon"
            obs_cond.attrs["long_name"] = (
                "Water level extremes conditional on precipitation extremes"
            )
            obs_cond.attrs["long_name_fr"] = (
                "Extrêmes de niveaux d'eau conditionnels aux extrêmes de précipitation"
            )
            obs_cond.attrs["description"] = "Dummy description"
            obs_cond.attrs["description_fr"] = "Description bidon"
    with xr.set_options(keep_attrs=True):
        obs_pot.attrs["ex_type"] = "Peaks over threshold"
        obs_cond.attrs["ex_type"] = "Conditional Extremes"
        obs_pot.attrs["stn_thresh"] = 5.0
        obs_pot.attrs["peaks_per_yr"] = 365 * (1 - 0.98)
    return obs_pot, obs_cond


def tas_obs():
    time = pd.date_range(start="1950-01-01", end="2020-12-31", freq="D")
    tas = np.random.randn(len(time), 1) * 50 + 273
    attrs = {
        "units": "K",
        "standard_name": "air_temperature",
        "long_name": "Temperature",
    }
    da = xr.DataArray(
        tas,
        coords={"time": time, "station": ["7028442"]},
        dims=["time", "station"],
        attrs=attrs,
        name="tas",
    )
    return da


def tas_sim():
    time = xr.cftime_range(start="1960-01-01", end="2100-12-31", freq="D")
    source_id = ["INM-CM4-8", "GFDL-CM4", "HadGEM3-GC31-MM", "MIROC6", "TaiESM1"]
    experiment_id = ["ssp126", "ssp245", "ssp370", "ssp585"]
    variant_label = ["r1i1p1f1", "r2i1p1f1"]
    station = ["7028442"]

    attrs = {
        "units": "K",
        "standard_name": "air_temperature",
        "long_name": "Temperature",
    }

    tas = (
        np.random.randn(
            len(time),
            len(station),
            len(source_id),
            len(experiment_id),
            len(variant_label),
        )
        * 50
        + 273
    )
    tas[365 * 50 :] += 1
    tas[:, 0, 0, 0, 0] = np.nan
    da = xr.DataArray(
        data=tas,
        coords={
            "time": time,
            "station": station,
            "source_id": source_id,
            "experiment_id": experiment_id,
            "variant_label": variant_label,
        },
        dims=["time", "station", "source_id", "experiment_id", "variant_label"],
        attrs=attrs,
        name="tas",
    ).stack(realization=("variant_label", "source_id", "experiment_id"))

    # Zarr won't store a multiindex.
    cn = ["variant_label", "source_id", "experiment_id"]
    realization = ["_".join(r) for r in da.realization.values]
    coords = {c: ("realization", da[c].values) for c in cn}
    coords["realization"] = realization

    da = da.drop_vars(coords.keys())
    return da.assign_coords(**coords)


if zarr.__version__.startswith("3"):
    zipclass = zarr.storage.ZipStore
else:
    zipclass = zarr.ZipStore


class MemoryZarrStore(zipclass):
    """
    store=MemoryZarrStore()
    ds.to_zarr(store=store)
    """

    def __init__(self):
        # store properties

        self.path = io.BytesIO()
        self.compression = zipfile.ZIP_STORED
        self.allowZip64 = True
        self.mode = "a"
        self._dimension_separator = None

        # Current understanding is that zipfile module in stdlib is not thread-safe,
        # and so locking is required for both read and write. However, this has not
        # been investigated in detail, perhaps no lock is needed if mode='r'.
        self.mutex = threading.RLock()

        # open zip file
        self.zf = zipfile.ZipFile(
            self.path,
            mode=self.mode,
            compression=self.compression,
            allowZip64=self.allowZip64,
        )
