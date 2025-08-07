import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

"""
Parse the IDF_Additional_Additionnel_v3-30/idfm0018.txt file from
https://collaboration.cmc.ec.gc.ca/cmc/climate/Engineer_Climate/IDF/idf_v3-30_2022_10_31/IDF_Files_Fichiers/
and extract annual maximum rainfall depths for different durations.

Notes
-----
Removed those entries (duplicated station numbers ans years... not sure which ones are valid)

11083802013  4.6  6.4  7.4 13.0 22.2 23.8 29.0 43.0 55.4VANCOUVER SEA ISLAND CCG                    TTTTTTTTT
11083802014  3.8  5.2  6.0  7.4  9.2 16.0 36.0 50.2 60.2VANCOUVER SEA ISLAND CCG                    TTTTTTTTT
11083802015  2.6  3.2  3.6  5.4  7.2 12.4 28.0 42.2 56.8VANCOUVER SEA ISLAND CCG                    TTTTTTTTT
11083802016  4.6  5.8  6.6  7.4 10.2 14.8 22.8 28.8 35.4VANCOUVER SEA ISLAND CCG                    TTTTTTTTT
11083802017 22.0 23.0 24.0 25.2 29.8 32.6 36.6 41.2 43.0VANCOUVER SEA ISLAND CCG                    TTTTTTTTT

10545032014-99.9-99.9-99.9-99.9 41.8 43.8 43.8 49.7 72.8LANGARA ISLAND RCS                          TTTTWWWWW
10545032015  4.0  5.8  7.4 10.2 12.0 15.2 31.0 53.4 64.4LANGARA ISLAND RCS                          TTTTTTTTT
10545032016 11.0 11.0 11.0 11.0 12.2 20.0 39.2 39.4 56.4LANGARA ISLAND RCS                          TTTTTTTTT
"""

# Duration in hours
duration_hr = {
    "5m": 1 / 12,
    "10m": 1 / 6,
    "15m": 1 / 4,
    "30m": 1 / 2,
    "1h": 1,
    "2h": 2,
    "6h": 6,
    "12h": 12,
    "24h": 24,
}


def parse(fh):
    """
    Generator parsing lines from a file handle.
    """
    f = np.float32

    # This is not pretty, but it's fast
    parse_line = lambda l: (
        l[0:7],
        int(l[7:11]),
        f(l[11:16]),
        f(l[16:21]),
        f(l[21:26]),
        f(l[26:31]),
        f(l[31:36]),
        f(l[36:41]),
        f(l[41:46]),
        f(l[46:51]),
        f(l[51:56]),
    )

    for line in fh:
        # Skip delimiter lines
        if len(line) > 11:
            yield parse_line(line)


def load_df(path: str | Path) -> pd.DataFrame:
    """Parse the IDF data and return a DataFrame indexed by station_id and year."""
    with open(path) as fh:
        raw = tuple(parse(fh))

    df = pd.DataFrame(
        raw,
        columns=[
            "station",
            "year",
            "5m",
            "10m",
            "15m",
            "30m",
            "1h",
            "2h",
            "6h",
            "12h",
            "24h",
        ],
    )
    return df.set_index(["station", "year"])


def load(path: str | Path) -> xr.Dataset:
    """Parse the IDF data and return xarray Dataset.

    Parameters
    ----------
    path : Path or str
      Path to the condensed IDF data file named `idfm0018.txt`.

    Returns
    -------
    xr.Dataset
      An xarray Dataset with annual maximum precipitation intensities for different rainfall durations.
    """
    # Load dataframe
    df = load_df(path)

    # Convert to xarray Dataset and convert units from mm to meters
    ds = xr.Dataset.from_dataframe(df.where(df != -99.9) / 1000)

    # Concat all durations into a single variable
    duration = xr.DataArray(list(ds.data_vars.keys()), dims="duration")
    da = xr.concat(ds.data_vars.values(), dim=duration)

    # Convert to intensity
    d = xr.DataArray(
        list(duration_hr.values()),
        dims="duration",
        coords={"duration": list(duration_hr.keys())},
    )
    da = da / d

    da.name = "idf"

    # Use datetime coordinates
    time = [dt.datetime(y, 12, 31) for y in ds.year.values]
    da["time"] = ("year", time)
    da = da.swap_dims(year="time").drop_vars("year")
    da.attrs = {
        "standard_name": "thickness_of_rainfall_amount",
        "long_name": "Annual maximum precipitation intensity for given duration",
        "units": "m/h",
    }

    # Add global attributes
    out = da.to_dataset()
    out.attrs = {
        "history": "Parsed from IDF_Additional_Additionnel_v3-30/idfm0018.txt. Masked values of -99.9. "
        "Converted from mm to m.",
        "source": "https://collaboration.cmc.ec.gc.ca/cmc/climate/Engineer_Climate/IDF/idf_v3-30_2022_10_31"
        "/IDF_Files_Fichiers/",
        "contact": "David Huard <huard.david@ouranos.ca>",
        "comments": "Data from duplicated station ids removed: 11083802013 and 10545032014",
        "description": "Annual precipitation maxima from ECCC",
    }

    return out


def load_meta(path):
    """Parse station metadata for IDF data that has been *included*."""
    df = pd.read_csv(
        path,
        sep=",",
        skiprows=1,
        skipinitialspace=True,
        names=[
            "station_id",
            "lat",
            "lon",
            "elev",
            "start",
            "end",
            "Yrs-Ans",
            "station_name",
            "prov",
        ],
        usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    # Remove stations with less than 20 years of data
    df = df[df["Yrs-Ans"] >= 20]
    df["valid_months"] = df["Yrs-Ans"] * 12
    df.drop("Yrs-Ans", axis=1, inplace=True)
    df["variable"] = "idf"
    df["station_name"] = df["station_name"].str.strip()
    df["start"] = [dt.datetime(int(y), 1, 1) for y in df["start"]]
    df["end"] = [dt.datetime(int(y), 12, 31) for y in df["end"]]

    header = [
        "station_id",
        "station_name",
        "lon",
        "lat",
        "elev",
        "variable",
        "valid_months",
        "start",
        "end",
    ]
    return df[header]


def save_idf_regions(path):
    """
    Load regions from INRS netCDF file, find the region for each IDF station, then save this region to a json file.

    """
    import json

    from scipy.spatial import KDTree

    from peach import frontend

    # Read region data from Guillaume Talbot
    ds = xr.open_dataset(path)
    ds = ds.rename_dims(Longitude="lon", Latitude="lat").rename_vars(
        longitude="lon", latitude="lat"
    )

    # Create KDTree for fast nearest neighbor search
    kd = KDTree(np.vstack([ds.lon.values, ds.lat.values]).T)

    # Read station data and pick IDF stations
    sv_path = (
        Path(__file__).parent.parent
        / "src"
        / "peach"
        / "frontend"
        / "data"
        / "stations_variables.csv"
    )
    df = pd.read_csv(sv_path)
    i = df["variable"] == "idf"

    # Identify region for each station by finding the nearest neighbor from each station
    d, k = kd.query(df[i][["lon", "lat"]])
    r = ds.region.isel(Region=k)

    # Save the region data to a json file
    out = dict(zip(df[i].station, map(int, r.values)))
    with open(Path(frontend.__file__).parent / "data" / "idf_regions.json", "w") as fh:
        json.dump(out, fh)

    # df.loc[i, "constraints"] = r.values.astype(int).astype(str)
    # df.to_csv(sv_path.with_suffix(".csv.mod"), index=False)


if __name__ == "__main__":

    def save(path: str | Path):
        """Save the IDF data to a zarr file."""
        ds = load(path)
        ds.to_zarr("backend/data/IDF3.30.zarr", mode="w")

    save(path=Path(__file__).parent.parent / "data" / "IDF3.30" / "idfm0018.txt")
    # meta = load_meta(path=Path(__file__).parent.parent / "data" / "IDF3.30" / "idf_v3-30_2022_10_31_log_included.txt")
