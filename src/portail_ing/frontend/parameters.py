"""
# Parameter classes for the panel application

These classes define the business logic of the application, but contain no visualization elements.

- Global: Global parameters, such as the locale.
- BaseParameterized: Base class for parameterized classes that need translation.
- Station: Station selection for each variable.
- GenericIndicator: A base class on which custom indice dependent Indicator classes will be created. This only holds the indicator metadata
- IndicatorArguments and IndexerIndicatorArguments : Template classes containing the arguments of a given indicator. They are dynamically created.
- IndicatorComputation: A container class to manage an indicator computation to analyse.
- IndicatorList: Container for the collection of indicators available to the user (GenericIndicator), and those that are selected for
  computation (IndicatorComputation).
- Analysis: Base class for handling all the analysis.
- IndicatorDA: A class to hold an indicator time series data for statistical analysis.
- IndicatorObsDA: Same but with distribution selection logic in the reference period.
- IndicatorRefDA: Same but
- IndicatorSimDA: Same but with simulation mixture logic in the future period.


abstract : description du calcul à faire
title : Nom court du calcul à faire
description : description du calcul fait
long_name : Nom court du calcul fait
"""

import calendar
import hashlib
import json
import traceback
from collections import defaultdict
from urllib.parse import urlencode, urlparse
from uuid import uuid4

import geopandas as gpd
import lmoments3 as lm3
import numpy as np
import pandas as pd
import param
import s3fs
import scipy
import xarray as xr
import xclim as xc
import xclim.ensembles
import xclim.indices.stats
from shapely.geometry import Point
from xclim.core.formatting import update_history
from xclim.core.units import units as xu

from portail_ing.common import config as global_config
from portail_ing.common.logger import get_logger
from portail_ing.common.request import AsyncJob, JobState, check_backend
from portail_ing.risk.priors import (
    members,
    model_weights_from_sherwood,
    scenario_weights_from_iams,
)
from portail_ing.risk.xmixture import XMixtureDistribution


def ddict():
    return defaultdict(ddict)


scipy_to_lmom = {
    "expon": "exp",
    "gamma": "gam",
    "genextreme": "gev",
    "genpareto": "gpa",
    "gumbel_r": "gum",
    "norm": "nor",
    "pearson3": "pe3",
    "weibull_min": "wei",
}
logger = get_logger(__name__)


# Supported locales
locale_map = {"en": "en_US.utf8", "fr": "fr_CA.utf8"}
# locales = list(locale_map.keys())
locales = {"Français": "fr", "English": "en"}
# locales = {"fr": "Français", "en": "English"}


# Allowed scipy distributions
# TODO: Update
scipy_dists = ["norm", "t", "gamma", "genextreme", "lognorm", "uniform"]

# Load the scenario weights as a function of time
scen_weights = scenario_weights_from_iams()

logger.info(f"Workspace: {global_config.WORKSPACE}")

# Mapping from dependent variables to core variables
VARIABLE_MAPPING = {"tasmax": "tas", "tasmin": "tas", "wl_pot": "wl"}

# Mapping for Parameter classes,  extended by idf_ and wl_parameters.
TYPE_MAP = {"obs": {}, "sim": {}}


class BaseParameterized(param.Parameterized):
    """Base class for parameterized classes that need translation."""

    # Language - pass ref from Global.locale
    locale = param.Selector(
        objects=locales, default="fr", allow_refs=True, precedence=-1
    )

    # Translation metadata for parameters' `label` and `doc` attributes, as well as values themselves
    _label = {}
    _doc = {}
    _value = param.Dict({}, instantiate=True, precedence=-1)

    @param.depends("locale", watch=True, on_init=True)
    def _translate_values_labels_docs(self):
        """Translate the indicator label and doc attributes when locale changes."""
        for key, obj in self.param.objects().items():
            # Translate the values
            if key in self._value:
                new = self._value[key].get(self.locale, None)
                if new:
                    setattr(self, key, new)

            # Translate the label and doc attributes
            for attr in ["label", "doc"]:
                tr = getattr(self, f"_{attr}")
                if key in tr:
                    if self.locale in tr[key]:
                        setattr(obj, attr, tr[key][self.locale])

    def label(self, attr):
        label = self._label.get(attr, attr)
        if isinstance(label, dict):
            label = label.get(self.locale, attr)

        return label

    def to_dict(self):
        """Return json representation of information from instance."""
        raise NotImplementedError


class URLParameterized(BaseParameterized):
    """View state, synchronized with the URL."""

    sync_on_params = {}
    sync_url = param.Boolean(True, precedence=-1)
    logger = get_logger("URLParameterized")
    ignored = {}
    _sync_callbacks = []
    _sync_watchers = []

    def __init__(self, sync_url=True, ignored_queries=[], sync_on_params={}, **kwargs):
        """Initialize URLParameterized

        Args:
            sync_url (param.Bool, optional): Param whether or not to sync all parameters to the URL
                (except those in ignored_queries, 'name', and 'sync_url'). Defaults to True.
            ignored_queries (list, optional): List of parameters to ignore in the sync/unsync operation.
                Defaults to [].
            sync_on_params (Dict[param.Parameterized:{str:str}], optional): Dict of param.Parameterized
                instances and their respective param names/values to watch to turn on/off URL sync.
                Defaults to {}.
        """
        super().__init__(**kwargs)
        self.logger = get_logger(f"URLParameterized.{self.name}")
        self.sync_url = sync_url
        self.ignored = {param: True for param in ignored_queries}
        self.ignored["name"] = True
        self.ignored["sync_url"] = True
        self.ignored["_value"] = True
        self.sync_on_params = sync_on_params
        self._sync_param_watcher = None
        self._sync_url_watcher = None
        self._is_syncing = False
        self.sync_init()

    def is_ignored(self, param):
        return (param in self.ignored) and (self.ignored.get(param, False))

    def sync_init(self):
        import importlib

        if importlib.util.find_spec("panel") is not None:
            import panel as pn

            if pn.state.location is not None:
                self.param.watch_values(self.set_sync_watchers, "sync_url", "value")
                pn.state.onload(self.sync_init_onload)

        else:
            self.sync_url = False

    def create_callback_sync_check_value(self, parameter, value):
        def callback(**kwargs):
            curr_value = kwargs.get(parameter)
            # self.logger.info(f'sync.callback, {kwargs}, Callback on {parameter} called with {curr_value}, checking if == {value}')
            if callable(value):
                compare = value(curr_value)
            elif isinstance(value, list) and not isinstance(curr_value, list):
                compare = curr_value in value
            else:
                compare = value == curr_value
            self.sync_url = compare

        return callback

    def init_check_to_sync(self):
        """Initialized watches for when sync_url should change, given self.sync_on_params

        Raises
        ------
            ValueError: Wrong type for key of sync_on_params.
            ValueError: Wrong type for key of sync_on_params[param]
        """
        for parameterized, param_dict in self.sync_on_params.items():
            if not isinstance(parameterized, param.Parameterized):
                raise ValueError(f"{parameterized} is not of type param.Parameterized")
            self.logger.info(
                f"{self.name}: Setting up watchers for {parameterized.name}"
            )
            for parameter, val in param_dict.items():
                if not isinstance(parameter, str):
                    raise ValueError(f"param {parameter} not a string (for .watch)")
                callback = self.create_callback_sync_check_value(parameter, val)
                self._sync_callbacks.append(callback)
                self._sync_watchers.append(
                    parameterized.param.watch_values(
                        callback, parameter, what="value", onlychanged=True, queued=True
                    )
                )
                callback(**{parameter: getattr(parameterized, parameter)})

    def url_unlisten(self):
        import panel as pn

        if self._sync_param_watcher is not None:
            self.param.unwatch(self._sync_param_watcher)
            self._sync_param_watcher = None

        if self._sync_url_watcher is not None:
            pn.state.location.param.unwatch(self._sync_url_watcher)
            self._sync_url_watcher = None

    def url_listen(self):
        params = self.param.values()
        valid_params = {
            key: val for key, val in params.items() if not self.is_ignored(key)
        }
        # unnecessary, changing the URL reloads the app in like 99.9 % of browsers...
        # self._sync_url_watcher = (
        #     pn.state.location.param.watch_values(
        #         self.sync_params_from_query,
        #         'search',
        #         'value'
        #     )
        # )
        self._sync_param_watcher = self.param.watch_values(
            self.sync_query_from_params,
            list(valid_params.keys()),
            "value",
        )

    def set_sync_watchers(self, sync_url=True, is_init=True):
        import panel as pn

        self.url_unlisten()
        params = self.param.values()
        valid_params = {
            key: val for key, val in params.items() if not self.is_ignored(key)
        }
        if sync_url:
            self.sync_params_from_query(search=pn.state.location.search)
            self.sync_query_from_params(**valid_params, only_new=True)
            self.url_listen()
        else:
            self.remove_from_url(valid_params.keys())

    def sync_init_onload(self):
        self.init_check_to_sync()
        try:
            self.set_sync_watchers(is_init=True)
        except Exception as err:
            logger.error(
                f"URL params sync of {self.__class__.__name__} object did not complete with : {err}"
            )

    def sync_query_from_params(self, **kwargs):
        import panel as pn

        """Callback function which is invoked when the parameters in valid_params change."""
        self.url_unlisten()

        only_new = kwargs.pop("only_new", False)
        update = False
        query = pn.state.location.query_params
        for p, v in kwargs.items():
            if only_new and p in query:
                continue
            if query.get(p) != v:
                # ensure strings in the url are parsed by ast.literal_eval as strings:
                if isinstance(v, str) and v[0] != '"' and v[-1] != '"':
                    v = f'"{v}"'
                query[p] = v
                update = True
        if update:
            pn.state.location.update_query(**query)

        self.url_listen()

    def sync_params_from_query(self, **kwargs):
        """
        Callback function which is invoked when the URL search parameter changes.
        """
        import panel as pn

        self.url_unlisten()

        params = self.param.values()
        valid_params = {
            key: val for key, val in params.items() if not self.is_ignored(key)
        }
        new_params = {}
        query = pn.state.location.query_params
        update = False
        for parameter, old_val in valid_params.items():
            if parameter in query:
                val = query[parameter]
                if val == "None":
                    val = None
                if isinstance(val, str) and val[0] == '"' and val[-1] == '"':
                    val = val[1:-1]
                if val != old_val:
                    update = True
                    new_params[parameter] = val
        if update:
            self.param.update(**new_params)
        self.url_listen()

    def remove_from_url(self, params):
        import panel as pn

        query = pn.state.location.query_params
        for parameter in params:
            if parameter in query:
                del query[parameter]
        pn.state.location.search = "?" + urlencode(query) if query else ""

    def on_error(self, unsyncable):
        self.logger.info(f"URL sync failed for {unsyncable}")


class Global(URLParameterized):
    """Global parameters."""

    locale = param.Selector(objects=locales, default="fr", precedence=10)
    backend = param.String(
        default=global_config.BACKEND_URL, constant=True, precedence=-1
    )
    tab = param.String(default="station_select", precedence=-1)
    sidebar_tab = param.List(default=[0])

    tabs = param.List(
        default=["station_select", "indicator_select"], constant=True, precedence=-1
    )
    _tab_to_ind = {}
    _ind_to_tab = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, tab in enumerate(self.tabs):
            self._tab_to_ind[tab] = i
            self._ind_to_tab[i] = tab
        if self.tab not in self._tab_to_ind:
            raise ValueError(f"Tab {self.tab} not in given tabs.")

    @param.depends("tab", watch=True)
    def on_change_tab(self):
        if self.tab not in self._tab_to_ind:
            raise ValueError(f"Tab {self.tab} not in given tabs.")

    @property
    def tab_index(self):
        return self.tab_to_ind(self.tab)

    def tab_to_ind(self, tab):
        return self._tab_to_ind[tab]

    def ind_to_tab(self, ind):
        return self._ind_to_tab[ind]


def latlng_to_proj(lat: float, lon: float):
    """Convert lat and lon to easting and northing."""
    # if lat is a scalar, convert to array:

    is_scalar = False
    if np.isscalar(lat) and np.isscalar(lon):
        is_scalar = True
        lat = [lat]
        lon = [lon]
    points = [Point(lon[i], lat[i]) for i in range(len(lat))]
    gpf = gpd.GeoDataFrame({"geom": points}, geometry="geom", crs="epsg:4326")
    gpf.to_crs(epsg=3857, inplace=True)
    geom = gpf.geometry.values

    if is_scalar:
        return geom[0].x, geom[0].y
    return geom.x, geom.y


def proj_to_latlng(easting: float, northing: float):
    """Convert easting and northing to lat and lon."""
    # if lat is a scalar, convert to array:
    is_scalar = False
    if np.isscalar(easting) and np.isscalar(northing):
        is_scalar = True
        easting = [easting]
        northing = [northing]
    points = [Point(easting[i], northing[i]) for i in range(len(easting))]
    gpf = gpd.GeoDataFrame({"geom": points}, geometry="geom", crs="epsg:3857")
    gpf.to_crs(epsg=4326, inplace=True)
    geom = gpf.geometry.values

    if is_scalar:
        return geom[0].x, geom[0].y
    # cool feature from shapely: don't need to unpack.
    return geom.x, geom.y


class Site(URLParameterized):
    # site variables:
    lat = param.Number(default=0, precedence=-1)
    lon = param.Number(default=0, precedence=-1)
    x = param.Number(default=0, precedence=-1)
    y = param.Number(default=0, precedence=-1)
    radius = param.Number(default=0, precedence=-1)
    enabled = param.Boolean(default=True, precedence=-1)
    is_updating = False

    @param.depends("lat", "lon", watch=True, on_init=True)
    def latlon_change(self):
        if not self.is_updating:
            x, y = latlng_to_proj(self.lat, self.lon)
            self.is_updating = True
            self.param.update(x=x, y=y)
            self.is_updating = False

    @param.depends("x", "y", watch=True)
    def proj_change(self):
        if not self.is_updating:
            lat, lon = proj_to_latlng(self.x, self.y)
            self.is_updating = True
            self.param.update(lat=lat, lon=lon)
            self.is_updating = False


class Map(URLParameterized):
    # bounds variables:
    clat = param.Number(default=0, precedence=-1)
    clon = param.Number(default=0, precedence=-1)
    z = param.Number(default=0, precedence=-1)


class Station(URLParameterized):
    """Station selector.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe with the station metadata. Should minimally include the columns station, station_name,
        lat, and lon, as well as one column per variable, with a boolean value indicating whether the variable is
        available at the station.
    variables : list of str
        The list of core variables available in the dataframe.
    station_id : dict
        A dictionary of the form {variable: station_id} indicating the selected station for each variable.
    """

    # List of core variables
    variables = param.Dict(default={}, precedence=-1, instantiate=True)

    # Convenience attribute providing a synthesis of selected station ID for each variable and its dependencies.
    station_id = param.Dict(default={}, instantiate=True, precedence=-1)
    site = param.ClassSelector(class_=Site)

    # Stations around site of interest
    site_df = param.Dict(
        default={},
        instantiate=True,
        precedence=-1,
        doc="Dict of str: dataframes, of key: variables -> val: stations around Site ",
    )
    selected_df = param.DataFrame(
        instantiate=True, precedence=-1, doc="Dataframe of selected stations."
    )

    # Note that the __init__ will create additional class attributes for each variable (tas, pr, etc)
    _label = {
        "pr": {"en": "Precipitations", "fr": "Précipitations"},
        "tas": {"en": "Temperature", "fr": "Température"},
        "wl": {"en": "Water level", "fr": "Niveau d'eau"},
        "idf": {"en": "Maximum rainfall", "fr": "Pluie maximale"},
        "station": {"en": "Station ID", "fr": "ID de la station"},
        "station_name": {"en": "Station name", "fr": "Nom de la station"},
        "lat": {"en": "Latitude", "fr": "Latitude"},
        "lon": {"en": "Longitude", "fr": "Longitude"},
        "elev": {"en": "Elevation", "fr": "Élévation"},
        "variable": {"en": "Variable", "fr": "Variable"},
        "valid_months": {"en": "# valid months", "fr": "# mois valides"},
        "start": {"en": "Start", "fr": "Début"},
        "end": {"en": "End", "fr": "Fin"},
        "distance": {"en": "Distance (km)", "fr": "Distance (km)"},
        "easting": {"en": "Easting", "fr": "Easting"},
        "northing": {"en": "Northing", "fr": "Northing"},
    }

    def __init__(self, df: pd.DataFrame, variables: list = [], **params):
        """Create a station selector for each variable.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe with the station metadata. Should minimally include the columns station, station_name,
            lat, and lon, as well as one column per variable, with a boolean value indicating whether the variable is
            available at the station.
        variables : list of str
            The list of variables available for which we want to identify stations. Defaults to all variables found
            in station metadata.
        """
        self._df = self._add_easting_northing(df)
        self._kdtree = {}
        super().__init__(**params)

        if self.variables == {}:
            unique_variables = pd.unique(self._df["variable"])
            # if given a subset of variables, pare down to this:
            if len(variables):
                unique_variables = [x for x in unique_variables if x in variables]
            for var in unique_variables:
                self.variables[var] = self._df[self._df["variable"] == var]
                self._kdtree[var] = scipy.spatial.KDTree(
                    self.variables[var][["easting", "northing"]].values
                )
        # Create a param.Selector for each variable, with the station names as options
        for var, sub in self.variables.items():
            objs = {"": None}
            objs.update(dict(zip(sub["station_name"], sub["station"])))

            # Add variable attribute to store the station_id selected by the user
            self.param.add_parameter(var, param.Selector(objects=objs))

            # Initialize
            self.station_id[var] = None
            for dep, core in VARIABLE_MAPPING.items():
                if core == var:
                    self.station_id[dep] = None

        self.param.watch(self._update_station_id_action, list(self.variables.keys()))
        self.param.watch(self.update_tables, ["station_id"])
        self.site.param.watch(self.update_tables, ["x", "y", "radius"])

    @staticmethod
    def _add_easting_northing(df: pd.DataFrame) -> pd.DataFrame:
        """Add easting and northing columns."""
        # Create Point geometry
        geom = df.apply(lambda row: Point(row.lon, row.lat), axis=1)

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs="epsg:4326")
        gdf.to_crs(epsg=3857, inplace=True)

        gdf["easting"] = gdf.apply(lambda row: row.geometry.x, axis=1)
        gdf["northing"] = gdf.apply(lambda row: row.geometry.y, axis=1)
        gdf["distance"] = np.nan
        gdf.drop(columns=["geometry"], inplace=True)
        return pd.DataFrame(gdf)

    # @param.depends("station_id", "site.radius", "site.x", "site.y",  watch=True, on_init=True)
    def update_tables(self, *events, **kwargs):
        logger.info(f"update_tables called with change in {[e.name for e in events]}")
        # check if any of the stations have changed, or the site has changed:
        easting = self.site.x
        northing = self.site.y
        radius = self.site.radius
        station_id = self.station_id

        changed_stations = []
        update_pos = False

        for e in events:
            if e.name == "station_id":
                old_stations = e.old
                changed_stations = [
                    k
                    for k in self.variables.keys()
                    if station_id.get(k) != old_stations.get(k)
                ]
            elif e.name in ["x", "y", "radius"] and e.old != e.new:
                update_pos = True

        station_dfs = {}
        with param.parameterized.batch_call_watchers(self):
            # check if any stations changed. If so, update selected_df.
            for var in changed_stations:
                s_id = station_id.get(var)
                if s_id is None:
                    continue
                check_replace_old = False
                if (
                    self.selected_df is not None and not self.selected_df.empty
                ) and self.selected_df[
                    (self.selected_df["station"] == s_id)
                    & (self.selected_df["variable"] == var)
                ].empty:

                    check_replace_old = True
                    old_ind = self.selected_df["variable"] == var
                    if old_ind.any():
                        self.selected_df.drop(
                            index=old_ind, errors="ignore", inplace=True
                        )

                if (
                    self.selected_df is None
                    or self.selected_df.empty
                    or check_replace_old
                ):
                    station_dfs[var] = self.variables[var][
                        (s_id == self.variables[var]["station"])
                    ]
                    if easting or northing:
                        station_dfs[var].loc[:, "distance"] = (
                            np.sqrt(
                                np.abs(station_dfs[var]["northing"] - northing) ** 2
                                + np.abs(station_dfs[var]["easting"] - easting) ** 2
                            )
                            / 1000
                        )

                    self.selected_df = pd.concat(
                        [station_dfs[var], self.selected_df], ignore_index=True
                    )

            # if station changed or site changed, update site_df
            if (len(changed_stations) and easting and northing and radius) or (
                update_pos
            ):

                # create tables
                for var in self.variables.keys():
                    sel = self.get_within(
                        var,
                        radius * 1000,
                        easting,
                        northing,
                    )
                    this_station_id = station_id.get(var)
                    if sel[sel.station == this_station_id].empty:
                        station_df = station_dfs.get(var)
                        if (
                            station_df is None
                            and self.selected_df is not None
                            and not self.selected_df.empty
                        ):
                            station_df = self.selected_df[
                                (self.selected_df.station == this_station_id)
                                & (self.selected_df.variable == var)
                            ]
                        if station_df is not None:
                            station_df.loc[:, "distance"] = (
                                np.sqrt(
                                    np.abs(station_df["northing"] - northing) ** 2
                                    + np.abs(station_df["easting"] - easting) ** 2
                                )
                                / 1000
                            )

                        self.site_df[var] = pd.concat(
                            [station_df, sel.reset_index(drop=True)], ignore_index=True
                        )
                    else:
                        self.site_df[var] = sel.reset_index(drop=True)

        self.param.trigger("site_df")
        self.param.trigger("selected_df")

    def distance_to_site(self, lat: float, lon: float):
        """Return the stations distance from a given point in km."""
        # Compute the distance to each station
        return haversine(lon, lat, self._df["lon"], self._df["lat"])

        # Sort the stations by distance
        # index = np.argsort(dist)
        # self.param[var].objects = {df.iloc[i]["station_name"]: df.iloc[i]["station"] for i in index}

    def _check_station_distance(self):
        """Return message if distance across stations exceed 50 km."""
        from scipy.spatial import distance_matrix

        coords = self.selected_df[["easting", "northing"]].values
        dist = distance_matrix(coords, coords).max() / 1000

        if dist > 50:
            if self.locale == "en":
                return (
                    f"Distance between selected stations is {dist:.0f} km. Note that this might be "
                    f"problematic for indicators relying on two variables measured at far away stations."
                )
            elif self.locale == "fr":
                return (
                    f"La distance entre les stations est de {dist:.0f} km. Notez que cela peut poser problème "
                    f"pour les indicateurs qui dépendent de deux variables mesurées à des stations éloignées."
                )

    def _update_station_id_action(self, *event):
        """Update data to reflect station data availability."""
        ids = {var: getattr(self, var) for var in self.variables}
        for dep, core in VARIABLE_MAPPING.items():
            if core in ids:
                ids[dep] = ids[core]

        self.station_id = ids

    # def _update_selected(self):
    #    """Store metadata for selected stations."""
    #    sid = [
    #        (self.station_id[v], v)
    #        for v in self.variables
    #        if self.station_id[v] is not None
    #    ]
    #    self.selected_df = (
    #        self._df.set_index(["station", "variable"]).loc[sid].reset_index()
    #    )

    def get_within(
        self,
        var,
        distance: float,
        easting: float = None,
        northing: float = None,
        p: int = 2,
        return_sorted: bool = True,
        **kwargs,
    ):
        """Get stations within a given distance (in metres) from the site."""
        # logger.info(f'get_within called with {distance}, {easting}, {northing}, {var}')
        points = self._kdtree[var].query_ball_point(
            tuple((easting, northing)),
            r=distance,
            p=p,
            return_sorted=return_sorted,
            **kwargs,
        )
        df = self.variables[var].iloc[points]
        if df.size:
            df.loc[:, "distance"] = (
                np.abs(df["northing"] - northing) ** p
                + np.abs(df["easting"] - easting) ** p
            ) ** (1 / p) / 1000
        return df

    @property
    def df(self):
        df = self._df.copy()
        # Compute distance from site
        # df["distance"] = self.distance_to_site(lat=self.lat, lon=self.lon)
        return df

    @property
    def vdf(self):
        return self.df.groupby("variable")

    def to_dict(self) -> dict:
        """
        Return
          - selected station metadata, and
          - site coordinates.
        """
        return {
            "station_id": self.station_id,
            "df": (
                self.selected_df.T.to_dict() if self.selected_df is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, station_id, df, site):
        """Create a new instance from a dictionary."""
        df = pd.DataFrame.from_dict(df).T
        out = cls(df=df, lat=site["lat"], lon=site["lon"], station_id=station_id)
        for key, val in station_id.items():
            setattr(out, key, val)
        return out


class GenericIndicator(BaseParameterized):
    """Generic description of an indicator and its availability for the selected stations.

    This is used to display the list of available indicators to the user.
    """

    locale = param.Selector(
        objects=locales, default="fr", precedence=-1, allow_refs=True
    )

    # Station ID for each variable - pass ref from Station.station_id
    station_id = param.Dict({}, precedence=-1, allow_refs=True)

    # Station metadata for the selected stations
    station_df = param.DataFrame(
        precedence=-1, doc="Selected station metadata.", allow_refs=True
    )

    # List of variables required for computation and whether they're in `station_id`
    variables = param.ListSelector(
        [], doc="Variables required for computation.", instantiate=True
    )
    # Argument config
    args = param.Dict({}, precedence=-1, instantiate=True)

    # xclim indicator identifier (registry ID)
    iid = param.String(default="", precedence=-1, instantiate=True, constant=True)

    # xclim indicator identifier
    identifier = param.String(
        default="", precedence=-1, instantiate=True, constant=True
    )

    title = param.String(default="", precedence=-1, instantiate=True)
    abstract = param.String(default="", precedence=-1, instantiate=True)

    # Backend process name for getting observed and simulated indicators
    obs_process = param.String(
        default="compute-indicators-obs", constant=True, precedence=-1
    )
    sim_process = param.String(
        default="compute-indicators-sim", constant=True, precedence=-1
    )

    _label = {
        "title": {"en": "Title", "fr": "Titre"},
        "abstract": {"en": "Abstract", "fr": "Résumé"},
    }

    def __init__(self, *args, required_variables, **kwargs):
        self.param.variables.objects = required_variables
        super().__init__(*args, **kwargs)

    @param.depends("locale", watch=True, on_init=False)
    def _translate(self):
        """Translate the indicator metadata when locale changes."""
        # Translate units
        for key, obj in self.param.objects().items():
            if key.endswith("_unit"):
                setattr(
                    self,
                    key,
                    xu.formatter.format_unit_babel(
                        xu(obj.default).units, spec="~", locale=self.locale
                    ),
                )

    @param.depends("station_id", watch=True, on_init=True)
    def _update_variables(self):
        """Set which variables are available for computation from the data."""
        vars = [
            k
            for (k, v) in self.station_id.items()
            if v is not None and k in self.param.variables.objects
        ]
        self.variables = vars

    @property
    def has_data(self) -> bool:
        """Return True if all variables required for computation have a station_id."""
        return len(self.variables) == len(self.param.variables.objects)

    @classmethod
    def from_xclim(cls, iid: str, config: dict, **kwargs):
        """Create an instance from an xclim indicator."""
        # Arguments and units parameters
        # Get metadata from the xclim indicator
        config = config or {}
        ind = xc.core.indicator.registry[iid].get_instance()
        meta = {"en": ind.json(), "fr": ind.translate_attrs("fr")}

        # New class attributes
        kwargs.update(identifier=ind.identifier, iid=iid)

        # Overwrite default backend process name if configured
        for k in ["obs_process", "sim_process"]:
            if k in config:
                kwargs[k] = config[k]

        # Translation metadata
        kwargs["_value"] = {
            key: {lang: meta[lang].get(key, "") for lang in locales.values()}
            for key in cls._label.keys()
        }

        # List of variables required for computation and whether they're in `station_id`
        variables = []
        for name, prm in ind.parameters.items():
            if prm.kind in [
                xc.core.utils.InputKind.VARIABLE,
                xc.core.utils.InputKind.OPTIONAL_VARIABLE,
            ]:
                variables.append(name)
        kwargs["required_variables"] = variables
        kwargs["args"] = config.get("args", {})

        return cls(**kwargs)


class IndicatorArguments(BaseParameterized):
    """Base class holding the indicator arguments."""

    # Backend option
    no_cache = param.Boolean(
        default=global_config.NO_CACHE_DEFAULT,
        precedence=-1,
        doc="Ask the backend to not use cached results.",
        instantiate=True,
        allow_refs=True,
    )

    @classmethod
    def from_xclim(cls, iid, config, locale, no_cache=global_config.NO_CACHE_DEFAULT):
        """Create a parametrized class holding the indicator arguments."""
        ind = xc.core.indicator.registry[iid].get_instance()

        # Create parameter objects for indicator arguments
        params = {"no_cache": no_cache}  # The Param objects
        doc = {}  # localized Doc for the param objects
        label = {}  # localized Labels for the param objects
        for name, prm in ind.parameters.items():
            if name not in config:
                continue

            conf = config[name]
            if prm.kind is xc.core.utils.InputKind.QUANTIFIED:
                # Default units from configuration
                du = conf["default_units"]

                # Convert default value from xclim
                default = xu.Quantity(prm.default).to(du)
                du_s = xu.formatter.format_unit_babel(
                    default.units, spec="~", locale="fr"
                )

                # The indicator parameter
                p = param.Number(
                    default=default.magnitude,
                    softbounds=(conf.get("vmin"), conf.get("vmax")),
                    doc=(
                        f"{prm.description} (vmin: "
                        f'{conf.get("vmin")} {du_s}, vmax: {conf.get("vmax")} {du_s})'
                    ),
                    precedence=10,
                    instantiate=True,
                    label=f"{name.replace('_', ' ').capitalize()} ({du_s})",
                )

                # Its unit
                unit = param.String(
                    default=conf["default_units"], label="Unit", precedence=-1
                )

                # Add parameter to compute indicator arguments
                params[name] = p
                params[name + "_unit"] = unit
            elif prm.kind is xc.core.utils.InputKind.NUMBER:
                params[name] = param.Number(
                    default=prm.default,
                    softbounds=(conf.get("vmin"), conf.get("vmax")),
                    doc=f"{prm.description} (vmin: {conf.get('vmin')}, vmax: {conf.get('vmax')})",
                    precedence=10,
                    instantiate=True,
                    label=name.replace("_", " ").capitalize(),
                )
            elif "choices" in conf:
                p = param.Selector(
                    objects=conf["choices"],
                    instantiate=True,
                    doc=prm.description,
                    label=name,
                )
                params[name] = p
            else:
                continue

            # Translation metadata for parameter
            doc[name] = {
                "fr": conf.get("doc", {}).get("fr", prm.description),
                "en": prm.description,
            }
            label[name] = {
                "fr": conf.get("label", {}).get("fr", p.label),
                "en": p.label,
            }

        if "indexer" in ind.parameters:
            bases = IndexingIndicatorArguments
        else:
            bases = cls

        kls = param.parameterized_class(f"{iid}Arguments", params, bases=bases)
        kls._label.update(label)
        kls._doc.update(doc)
        return kls(locale=locale)

    def set_values(self, params):
        """Set the parameter values from a dict."""
        for key, val in params.items():
            if key not in self.param:
                raise ValueError(f"{key} is not a valid argument for this indicator.")
            if (keyunit := f"{key}_unit") in self.param:
                q = xu(val)
                setattr(self, key, q.m)
                setattr(self, keyunit, str(q.units))
            else:
                setattr(self, key, val)


class IndexingIndicatorArguments(IndicatorArguments):
    """Frequency and indexer indicator parameters.

    We assume that freq is always annual, but we can tweak the indexer.
    """

    # Freq options
    start_m = param.Selector(
        objects={}, default=1, label="Start month", precedence=20, instantiate=True
    )
    end_m = param.Selector(
        objects={}, default=12, label="End month", precedence=21, instantiate=True
    )

    # Months abbreviations hardcoded to avoid locale installation issues
    _label = {
        "start_m": {"en": "Start month", "fr": "Mois de début"},
        "end_m": {"en": "End month", "fr": "Mois de fin"},
        "months": {
            "en": [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            "fr": [
                "Jan",
                "Fév",
                "Mar",
                "Avr",
                "Mai",
                "Jun",
                "Jui",
                "Aoû",
                "Sep",
                "Oct",
                "Nov",
                "Déc",
            ],
        },
    }

    _base = "YS"

    @param.depends("locale", watch=True, on_init=True)
    def _translate_months(self):
        """Translate the month names."""
        months = {m: i + 1 for i, m in enumerate(self._label["months"][self.locale])}
        self.param["start_m"].objects = months
        self.param["end_m"].objects = months

    @property
    def freq(self):
        """Return the frequency string."""
        return f"{self._base}-{calendar.month_abbr[self.start_m].upper()}"

    @property
    def indexer(self):
        """Return the indexer dictionary."""
        # Indexes of start and end months
        ii = list(range(1, 13)) + list(range(1, 13))
        i1 = ii.index(self.start_m)
        i2 = ii.index(self.end_m, i1)

        # List of included months
        mi = np.arange(i1, i2 + 1) % 12 + 1

        if len(mi) == 12:
            return {}
        else:
            return {"month": mi.tolist()}

    def set_values(self, params):
        """Set the parameter values from a dict.

        Understands the indexer argument, but only for months.
        """
        if "indexer" in params and params["indexer"]:
            if "month" not in params["indexer"] or len(params["indexer"]) != 1:
                raise ValueError(
                    f"The indexer argument only supports a month list. Got : {params['indexer'].keys()}"
                )
            self.start_m = params["indexer"]["month"][0]
            self.end_m = params["indexer"]["month"][-1]
        super().set_values(
            {k: v for k, v in params.items() if k not in ["indexer", "freq"]}
        )


class IndicatorComputation(BaseParameterized):
    """An indicator ready for computation."""

    base = param.ClassSelector(class_=GenericIndicator)

    # Unique id for the indicator instance and its parameters
    # This allows distinguishing between different instances of the same indicator
    uuid = param.String(default="", instantiate=True, constant=True, precedence=-1)

    # True if already computed.
    computed = param.Boolean(False, precedence=-1)

    # Arguments
    args = param.ClassSelector(class_=IndicatorArguments, precedence=-1)

    obs_job = param.ClassSelector(class_=AsyncJob)
    sim_job = param.ClassSelector(class_=AsyncJob)

    @classmethod
    def from_generic(cls, base, no_cache=global_config.NO_CACHE_DEFAULT):
        """Create an Indicator instance from a GenericIndicator."""
        args = IndicatorArguments.from_xclim(
            base.iid, base.args, locale=base.param.locale, no_cache=no_cache
        )

        return cls(
            args=args,
            base=base,
            locale=base.param.locale,
            uuid=str(uuid4()),
            obs_job=AsyncJob(max_retries=global_config.MAX_RETRIES),
            sim_job=AsyncJob(max_retries=global_config.MAX_RETRIES),
        )

    def to_dict(self) -> dict:
        """Return dictionary with all indicator compute parameters, except `ds`."""
        # Dict for parameters
        values = self.args.param.values()
        if isinstance(self.args, IndexingIndicatorArguments):
            values.update(freq=self.args.freq, indexer=self.args.indexer)

        # xclim instance
        ind = xc.core.indicator.registry[self.base.iid].get_instance()

        p = {}
        for key in ind.parameters.keys():
            if key + "_unit" in values:
                # QUANTIFIED: Combine the value and unit in a string
                p[key] = f"{values[key]} {values[key + '_unit']}"
            elif key in values:
                p[key] = values[key]

        return {"name": self.base.iid, "params": p}

    @property
    def hash(self) -> str:
        """Return a hash of the indicator and its parameters."""
        s = json.dumps(self.to_dict()).encode()
        return hashlib.md5(s).hexdigest()

    def post_request(self, backend, repost=False):
        """Post the compute request to the given backend.

        If repost is False, only post if job state is unsent, submit failed or failed
        If True, post no matter what.
        """
        if not self.base.has_data:
            raise ValueError("Missing station data for some variables.")

        # Add the station information to the input data.
        data = self.to_dict() | {
            "stations": {
                var: str(self.base.station_id[var]) for var in self.base.variables
            }
        }

        if repost or self.obs_job.state <= JobState.failed:
            self.obs_job.post(backend, self.base.obs_process, data={"inputs": data})
        if repost or self.sim_job.state <= JobState.failed:
            self.sim_job.post(backend, self.base.sim_process, data={"inputs": data})


class IndicatorList(BaseParameterized):
    """Container for the collection of indicators available to the user, and those that are selected for computation."""

    # The Indicator classes of all available indicators
    indicators = param.Dict(default={}, instantiate=True)

    # The selected indicators (uuid, indicator instance)
    selected = param.Dict(default={}, precedence=1, instantiate=True)

    # True when all selected indicators are computed
    allcomputed = param.Boolean(False, precedence=-1)

    # The station ID for each variable - pass ref from Station.station_id
    station_id = param.Dict(
        default={}, precedence=-1, allow_refs=True, instantiate=True
    )

    # The configuration for each indicator
    config = param.Dict(default={}, precedence=-1, instantiate=True)

    # Backend URL for climate indicator computations
    backend = param.String(allow_refs=True, precedence=-1, instantiate=True)

    # Backend option
    no_cache = param.Boolean(
        default=global_config.NO_CACHE_DEFAULT,
        precedence=-1,
        doc="Ask the backend to not use cached results.",
        instantiate=True,
        allow_refs=True,
    )

    # Link to results
    result_links = param.Dict(
        default={"obs": {}, "sim": {}},
        precedence=-1,
        doc="Links to the indicators' computation.",
        instantiate=True,
    )

    # Used in sub-section titles
    _label = {
        "indicators": {"en": "Available indicators", "fr": "Indicateurs disponibles"},
        "selected": {"en": "Selected indicators", "fr": "Indicateurs sélectionnés"},
    }

    def __init__(self, **kwargs):
        """Create a new Indicators object.

        Combine information from the configuration and from the xclim indicator metadata.
        """
        super().__init__(**kwargs)

        # Populate the indicators available for computation
        for iid, conf in self.config.items():
            self.indicators[iid] = GenericIndicator.from_xclim(
                iid,
                config=conf,
                locale=self.param.locale,
                station_id=self.param.station_id,
            )

    def add(self, iid: str):
        """Add an indicator to the selected indicators."""
        if len(self.selected) == global_config.MAX_INDICATORS:
            raise OverflowError(
                f"Maximal number of indicators reached : {global_config.MAX_INDICATORS}."
            )

        ind = IndicatorComputation.from_generic(
            self.indicators[iid], no_cache=self.param.no_cache
        )
        self.selected[ind.uuid] = ind
        self.param.trigger("selected")
        return ind.uuid

    def remove(self, uuid: str):
        """Remove an indicator from the selected indicators."""
        self.selected.pop(uuid)
        self.param.trigger("selected")

    def ping_backend(self):
        """Check that the backend server is running."""
        return check_backend(self.backend)

    def post_all_requests(self, repost=False, wait=True):
        for ind in self.selected.values():
            if not ind.computed:
                ind.post_request(self.backend, repost)

    def monitor_jobs(self, event=None):
        """Get the progress of the computationss if completed, get the result link and load results.

        If the computation failed, try it again until the maximum number of retries is reached.
        """
        n_active = 0
        links_update = False
        for ind in self.selected.values():
            if ind.computed:
                continue

            oj = ind.obs_job.monitor()
            if oj in [JobState.accepted, JobState.in_progress]:
                n_active += 1
            elif oj is JobState.successful:
                self.result_links["obs"][ind.uuid] = ind.obs_job.result
                links_update = True

            sj = ind.sim_job.monitor()
            if sj in [JobState.accepted, JobState.in_progress]:
                n_active += 1
            elif sj is JobState.successful:
                self.result_links["sim"][ind.uuid] = ind.sim_job.result
                links_update = True

            if (
                not ind.computed
                and oj is JobState.successful
                and sj is JobState.successful
            ):
                ind.computed = True

        is_done = False
        if n_active == 0:
            logger.debug("No active jobs left. Stopping monitoring.")
            is_done = True
        else:
            logger.info(f"Number of active jobs remaining: {n_active}")

        if links_update:
            self.param.trigger("result_links")
        return is_done

    @param.depends("selected", "result_links", watch=True, on_init=True)
    def check_all_computed(self, event=None):
        allcomp = all([ind.computed for ind in self.selected.values()]) and (
            len(self.selected) > 0
        )
        if allcomp != self.allcomputed:
            self.allcomputed = allcomp

    def select_from_list(self, data):
        """Selects indicators from a list of dicts.

        Dicts can have 2 entries:
            - name : the indicator xclim id (required)
            - params : a dict of xclim arguments (optional)

        Returns a list of UUIDs in the same order as given.
        """
        uuids = [self.add(ind["name"]) for ind in data]
        for uuid, ind in zip(uuids, data):
            self.selected[uuid].args.set_values(ind.get("params", {}))
        self.param.trigger("selected")
        return uuids

    def to_dict(self, mode="results"):
        """Return selected indicators and results links.

        If mode is 'request', this returns a list, without UUIDs.
        """
        if mode == "results":
            inds = {uuid: ind.to_dict() for uuid, ind in self.selected.items()}
        elif mode == "request":
            uuids = list(sorted(self.selected.keys()))
            inds = [self.selected[uuid].to_dict() for uuid in uuids]
        return inds


class Analysis(BaseParameterized):
    """Class for indicators analysis."""

    indicators = param.ClassSelector(
        class_=IndicatorList, precedence=1, instantiate=True
    )

    # Indicator dataset - DataArrays are keyed by UUID
    ds = param.Parameter(
        default={"obs": {}, "sim": {}}, doc="Indicator datasets", instantiate=True
    )

    ref_period = param.Range(
        (1990, 2020), bounds=(1900, 2100), precedence=10, instantiate=True
    )
    fut_period = param.Range(
        (2030, 2060), bounds=(2025, 2100), precedence=10, instantiate=True
    )

    # Individual indicators, keyed by indicator
    obs = param.Dict({})
    ref = param.Dict({})
    fut = param.Dict({})
    checked = ddict()

    level = param.Number(0.05, doc="KS test significance level")

    station_df = param.DataFrame(
        precedence=-1,
        doc="Selected station metadata.",
        allow_refs=True,
    )

    # Configuration
    conf = param.Dict({}, precedence=-1)

    _label = {
        "ref_period": {"en": "Reference period", "fr": "Période de référence"},
        "fut_period": {"en": "Future period", "fr": "Période future"},
    }

    def load_da(self, link):
        """Load DataArray from disk.

        Converts data from dims (realization, time) to (variant_label, source_id, experiment_id, time).
        """
        if str(link).startswith("http"):
            # minio / s3fs
            url = urlparse(str(link))
            s3r = s3fs.S3FileSystem(
                anon=True,
                use_ssl=url.scheme == "https",
                endpoint_url=f"{url.scheme}://{url.netloc}",
            )
            link = url.path
            store = s3r.get_mapper(link, check=False)
            return xr.open_dataarray(store, decode_timedelta=False, engine="zarr")
        # else
        out = xr.open_dataarray(
            global_config.WORKSPACE / link, decode_timedelta=False, engine="zarr"
        )
        return out

    @param.depends("ref_period", watch=True)
    def watch_ref_period(self):
        logger.info(f"watch_ref_period: Triggered with {self.ref_period}")

        update_checks = {}
        for kind in self.ds.keys():
            for uuid in self.ds[kind]:
                da = self.ds[kind][uuid]
                self.perform_checks(uuid, kind, da, update_checks, force=True)
        self.param.update(**update_checks)

    @param.depends("indicators.result_links", watch=True, on_init=True)
    def _load_results(self, links=None):
        """Load the results from the backend."""
        if links is None:
            if self.indicators is not None:
                links = self.indicators.result_links
            else:
                return
        updated = False
        update_checks = {}

        for kind in ["obs", "sim"]:
            for uuid, link in links[kind].items():
                if uuid not in self.ds[kind]:
                    da = self.load_da(link)
                    self.perform_checks(uuid, kind, da, update_checks)
                    self.ds[kind][uuid] = da
                    updated = True
            if updated:
                logger.debug(f"Loading results: {kind}")
                self.param.update(**update_checks)
                self.param.trigger("ds")

    def perform_checks(self, uuid, kind, da, updaters: dict = {}, force=False):
        """Performs data checks to data array to ensure params are valid

        Parameters
        ----------
        uuid : str
            uuid of indicator
        kind : str
            type of data (obs, ref)
        da : xarray.DataArray
            Data array of indicator data, indexed uniquely by time.
        updaters : dict, optional
            argument to hold output, by default {}, to be passed to param.update

        Returns
        -------
        dict
            updaters; updated with appropriate values.
        """
        if (kind == "obs") and (force or not self.checked[uuid][kind]["ref_period"]):
            ref_period = updaters.get("ref_period", self.ref_period)
            new_period = self.check_extend_ref_period(da, ref_period)

            updaters["ref_period"] = (
                min(ref_period[0], new_period[0]),
                max(ref_period[1], new_period[1]),
            )

            self.checked[uuid][kind]["ref_period"] = True

        return updaters

    def check_extend_ref_period(self, da, ref_period: tuple) -> tuple:
        """Checks if ref_period is valid for the data array, and if not returns a new ref_period which is extended to ensure it is valid.

        Parameters
        ----------
        da : xarray.DataArray
            data array indexed by time, potentially with nans.
        ref_period : tuple
            2-tuple of (start_year, end_year), as ints.

        Returns
        -------
        tuple
            updated 2-tuple of (start_year, end_year), as ints.
        """
        years_with_obs_data = da.where(np.isfinite(da), drop=True).time
        years_in_per = years_with_obs_data.sel(time=slice(*map(str, ref_period)))
        count = years_in_per.count().item()

        if count >= global_config.MIN_OBS_DATA:
            return ref_period
        else:

            new_ref_period = self.extend_range_to_arr(
                years_with_obs_data.dt.year.data, global_config.MIN_OBS_DATA, ref_period
            )
            logger.debug(
                f"Extending Reference Period: {ref_period} to {new_ref_period}"
            )
            return new_ref_period

    def extend_range_to_arr(self, arr: np.array, n: int, r: tuple) -> tuple:
        """Finds the closest range with `n` valid elements in an array `arr`, for the tuple range `r`

        Parameters
        ----------
        arr : np.array
            array to index
        n : int
            number of elements to include (if n > arr.size, then this function uses n = arr.size without warning.)
        r : tuple
            2-tuple of range of years as integers

        Returns
        -------
        tuple
            updated 2-tuple of range of years as integers, with at least n valid elements.

        Notes
        -----
        Runs in O(arr.size) time, in the worst case.
        """

        low = r[0]
        high = r[1]
        # can't get more than arr.size...

        s = min(n, arr.size) - 1

        x = np.zeros_like(arr)
        x[arr < low] = low - arr[arr < low]
        x[arr > high] = arr[arr > high] - high

        # first s elements, in any order:
        p = np.argpartition(x, s)[:s]
        closest_n_years = arr[p]

        low = min(low, np.min(closest_n_years).item())
        high = max(high, np.max(closest_n_years).item())

        return (low, high)

    @param.depends("ds", watch=True, on_init=True)
    def _set_ref_period_bounds(self):
        """Set the reference period bounds from the dataset."""
        # TODO: move to perform_checks
        for uuid, da in self.ds["obs"].items():
            years = da.time.dt.year
            start = int(years.isel(time=0).values)
            end = int(years.isel(time=-1).values)

            self.param.ref_period.bounds = (start, end)
            break

    @param.depends("ds", watch=True, on_init=True)
    def _set_fut_period_bounds(self):
        """Set the reference period bounds from the dataset."""
        # TODO: move to perform_checks
        for uuid, da in self.ds["sim"].items():
            years = da.time.dt.year
            start = int(years.isel(time=0).values)
            end = int(years.isel(time=-1).values)

            self.param.fut_period.bounds = (start, end)
            break

    @param.depends("ds", watch=True, on_init=True)
    def _update_obs(self):
        """Update the observed indicators available for analysis."""
        update = False

        # Cast generator to list to avoid "dictionary changed size during iteration" errors
        for uuid, da in list(self.ds["obs"].items()):
            # Get the appropriate class for the indicator (IDF, WL or standard)
            kls = TYPE_MAP["obs"].get(da.name, IndicatorObsDA)

            # Set the station metadata if provided
            if self.station_df is not None:
                core_vars = [
                    VARIABLE_MAPPING.get(key, key)
                    for key in da.attrs["stations"].keys()
                ]
                station_df = self.station_df.set_index("variable").loc[core_vars]
            else:
                station_df = None

            # Instantiate the indicator class
            # That that here we change the `name` of the DataArray to the UUID, so Panel doesn't synchronize the y-axes
            # Don't move this up into `load_results` otherwise the TYPE_MAP won't work
            if uuid not in self.obs:
                logger.debug("Updating analysis OBS with %s", uuid)
                self.obs[uuid] = kls.from_da(
                    da=da,
                    period=self.param.ref_period,
                    name=uuid,
                    station_df=station_df,
                    locale=self.param.locale,
                )
                update = True

        if update:
            self.param.trigger("obs")

    @param.depends("ds", "_update_obs", watch=True, on_init=True)
    def _update_sim(self):
        """Update the observed indicators available for analysis."""
        update = False
        try:
            # Cast generator to list to avoid "dictionary changed size during iteration" errors
            for uuid, da in list(self.ds["sim"].items()):
                kls_ref, kls_fut = TYPE_MAP["sim"].get(
                    da.name, (IndicatorRefDA, IndicatorSimDA)
                )

                if uuid not in self.ref and uuid in self.obs:
                    self.ref[uuid] = kls_ref.from_da(
                        da=da,
                        name=uuid,
                        dist=self.obs[uuid].param.dist,
                        obs=self.obs[uuid],
                        level=self.param.level,
                        station_df=self.obs[uuid].station_df,
                        locale=self.param.locale,
                    )
                    self.fut[uuid] = kls_fut.from_da(
                        da=da,
                        period=self.param.fut_period,
                        name=uuid,
                        dist=self.obs[uuid].param.dist,
                        obs=self.obs[uuid],
                        model_weights=self.ref[uuid].param.model_weights,
                        locale=self.param.locale,
                    )
                    update = True

            if update:
                self.param.trigger("ref")
                self.param.trigger("fut")
        except Exception as err:
            logger.error(f"update_sim - Got error {err}.")
            print(traceback.format_exc())
            raise

    @param.depends("indicators.selected", watch=True)
    def maybe_remove_indicators(self, event=None):
        #!FIXME This only removes the indicators from the views, the computations are still triggered somehow.
        extras = set(self.obs.keys()) - set(self.indicators.selected.keys())
        if extras:
            logger.info(f"Removing indicators {extras}")
        for extra in extras:
            del self.ds["obs"][extra]
            del self.ds["sim"][extra]
            del self.obs[extra]
            del self.ref[extra]
            del self.fut[extra]
        if extras:
            self.param.trigger("obs")
            self.param.trigger("ref")
            self.param.trigger("fut")

    def update_params(self, ref_period=None, fut_period=None, metric=None, dist=None):
        """Update the periods, distributions and metrics.

        Distributions and metrics are mappings from indicator uuid to value.
        """
        # TODO ? Sadly, when this function updates both a dist and a metric, multiple triggers are sent consecutively.
        if metric is not None:
            for uuid, met in metric.items():
                if met is not None and self.obs[uuid].metric != met:
                    self.obs[uuid].metric = met
        if dist is not None:
            for uuid, dst in dist.items():
                if dist is not None and self.obs[uuid].dist != dst:
                    self.obs[uuid].dist = dst

        updates = {}
        if ref_period is not None and self.ref_period != tuple(ref_period):
            updates["ref_period"] = tuple(ref_period)
        if fut_period is not None and self.fut_period != tuple(fut_period):
            updates["fut_period"] = tuple(fut_period)
        if updates:
            self.param.update(**updates)

    def to_dict(self, mode="results"):
        """Return a dict representation of the class.

        If full is True, all data needed to reconstruct the object is included in the output.
        If it is False, only information directly relevant (periods, names, chosen distributions and metrics) to the results is included.
        """
        if mode == "results":
            return {
                "ref_period": self.ref_period,
                "fut_period": self.fut_period,
                "level": self.level,
                "observed": {key: ind.to_dict() for key, ind in self.obs.items()},
                "reference": {key: ind.to_dict() for key, ind in self.ref.items()},
                "future": {key: ind.to_dict() for key, ind in self.fut.items()},
            }
        uuids = list(sorted(self.obs.keys()))
        if mode == "short":
            return {
                "ref_period": self.ref_period,
                "fut_period": self.fut_period,
                "level": self.level,
                "indicators": [self.obs[uuid].to_dict(mode="short") for uuid in uuids],
            }
        if mode == "request":
            return {
                "ref_period": self.ref_period,
                "fut_period": self.fut_period,
                "dist": [self.obs[uuid].dist for uuid in uuids],
                "metric": [getattr(self.obs[uuid], "metric", None) for uuid in uuids],
            }


class IndicatorDA(BaseParameterized):
    """Indicator DataArray analysis parameters.

    Notes
    -----
    data : data read from backend output
    ts : time series to be displayed
    sample: data for the statistical analysis over the selected period
    """

    long_name = param.String("", doc="Long name of indicator", instantiate=True)
    description = param.String("", doc="Indicator description", instantiate=True)
    xid = param.String("", doc="xclim indicator call", instantiate=True)

    # Default is first entry in the list
    dist = param.Selector(
        default=None,
        objects=scipy_dists,
        label="Distribution",
        doc="Statistical distribution",
        allow_refs=True,
        instantiate=True,
    )

    period = param.Range(
        allow_refs=True,
        doc="Period for statistical analysis",
        allow_None=False,
        instantiate=True,
    )

    station_df = param.DataFrame(
        precedence=-1,
        doc="Station metadata.",
        allow_refs=True,
        instantiate=True,
    )

    data = param.Parameter(
        doc="DataArray time series from the backend computation",
        constant=True,
        instantiate=True,
    )
    ts = param.Parameter(
        doc="DataArray time series for public display", constant=False, instantiate=True
    )

    dparams = param.Parameter(doc="Distribution parameters", instantiate=True)

    _label = {
        "long_name": {"en": "Title", "fr": "Titre"},
        "description": {"en": "Description", "fr": "Description"},
        "period": {"en": "Period", "fr": "Période"},
    }

    @classmethod
    def from_uri(cls, uri, **kwargs):
        """Create an instance from a URI."""
        return cls.from_da(da=xr.open_dataarray(uri), **kwargs)

    @classmethod
    def from_da(cls, da: xr.DataArray, **kwargs):
        """Create an instance from a DataArray."""
        # Translations
        tr = {
            "long_name": {"en": da.attrs["long_name"], "fr": da.attrs["long_name_fr"]},
            "description": {
                "en": da.attrs["description"],
                "fr": da.attrs["description_fr"],
            },
        }
        xid = xid_from_da(da)

        out = cls(data=da, xid=xid, _value=tr, **kwargs)
        # out.param.trigger("locale")
        return out

    @param.depends("data", watch=True, on_init=True)
    def _update_ts(self):
        """Set the time series to be displayed."""
        self.ts = self.data

    def _dist_method(self, name, arg: xr.DataArray | float):
        # if not isinstance(arg, (float, int, xr.DataArray)):
        #     # The dimension must match that used by XMixtureDistribution
        #     arg = xr.DataArray(data=arg, dims="point")
        with xr.set_options(keep_attrs=True):
            out = xc.indices.stats.dist_method(
                function=name, fit_params=self.dparams, arg=arg
            )
        out.name = name
        return out

    def _slice(self, period: (int, int)) -> slice:
        """Return the reference period as a slice."""
        return slice(*map(str, period))

    def _sample(self, period: tuple) -> xr.DataArray:
        """Return the data during the period."""
        return self.ts.sel(time=self._slice(period))

    @property
    def sample(self) -> xr.DataArray:
        """Return the data during the period."""
        return self._sample(self.period)

    def fit(self, dist: str, period: tuple, method="PWM", iteration=0) -> xr.DataArray:
        """Fit the distribution to the data."""
        sample = self._sample(period)
        if method is None or method == "PWM":
            lmom_dist = scipy_to_lmom.get(dist, dist)
            if hasattr(lm3.distr, lmom_dist):
                dist_obj = getattr(lm3.distr, lmom_dist)
            else:
                method = "ML"
                dist_obj = dist
        else:
            dist_obj = dist

        logger.info(f"lmom: {dist}, {dist_obj} {method}")
        res = None
        try:
            res = xc.indices.stats.fit(sample, dist=dist_obj, dim="time", method=method)
        except Exception as e:
            if iteration < 1:
                logger.warning(
                    f"fit encountered error: {e}. Trying again with method=ML"
                )
                return self.fit(dist, period, method="ML", iteration=(iteration + 1))
            else:
                raise e

        logger.info(f"fit: {res.data} ")
        return res

    def pdf(self, x):
        """Return the probability density function."""
        return self._dist_method("pdf", x)

    def cdf(self, value) -> float:
        """Cumulative Distribution Function (CDF)."""
        return self._dist_method("cdf", value)

    def ppf(self, value) -> float:
        """Percent Point Function, inverse of CDF."""
        return self._dist_method("ppf", value)

    def sf(self, value) -> float:
        """Survival function (1-CDF), or probability of exceedance."""
        return self._dist_method("sf", value)

    def isf(self, value):
        """Inverse survival function."""
        return self._dist_method("isf", value)

    @param.depends("dist", "period", "ts", watch=True, on_init=True)
    def _update_params(self):
        """Update the distribution parameters."""
        if self.dist is not None and self.ts is not None:
            with param.parameterized.discard_events(self):
                self.dparams = self.fit(self.dist, self.period)
        self.param.trigger("dparams")

    def to_dict(self, mode="results"):
        """Return a dict representation of the class."""
        out = {
            "long_name": self.long_name,
            "distribution": self.dist,
        }
        if mode == "results":
            out.update(
                {
                    "period": self.period,
                    "parameters": self.dparams.to_dict(),
                    "data": self.sample.to_dict(),
                }
            )
        return out


class IndicatorObsDA(IndicatorDA):
    """Observed indicator DataArray analysis parameters.

    Includes logic to pick the best distribution.
    """

    # Information criteria metrics
    metric = param.Selector(
        objects=["aic", "bic"], default="bic", doc="Information criterion"
    )

    # Metric values for all distributions - DataArrays
    metrics = param.Dict({"aic": None, "bic": None})

    @param.depends("dist", "period", watch=True, on_init=True)
    def _update_params(self):
        """Update the distribution parameters."""
        if self.dist is None:
            self.dist = self.best_dist()
        # trigger change even if nothing changes:
        with param.parameterized.discard_events(self):
            self.dparams = self.fit(self.dist, self.period)
        self.param.trigger("dparams")

    @param.depends("metrics", watch=True)
    def _update_dist_selector(self):
        """Update the distribution selector with metric values."""
        m = self.metrics[self.metric]

        if m is not None:
            objs = {
                f"{k} ({self.metric.upper()}: {v:.2f})": k
                for k, v in sorted(m.items(), key=lambda x: x[1])
            }
            self.param.dist.objects = objs

    def _ll(self, params, sample) -> xr.DataArray:
        """Return the log-likelihood of the distribution."""
        # TODO: Replace by dist_method to shrink code base ?
        from portail_ing.risk.mixture import parametric_logpdf as logpdf

        return logpdf(params, v=sample).sum(dim="logpdf")

    def _bic(self, dist: str, period: tuple) -> xr.DataArray:
        """Return the Bayesian Information Criterion.

        BIC = log(n) k - 2 log(L)
        """
        sample = self._sample(period)
        dparams = self.fit(dist, period)
        ll = self._ll(dparams, sample)
        out = np.log(len(sample)) * len(dparams) - 2 * ll
        out.attrs = {
            "long_name": "Bayesian Information Criterion",
            "description": "BIC = log(n) k - 2 log(L)",
            "history": update_history(
                "BIC", new_name="bic", parameters=dparams, sample=sample
            ),
            "scipy_dist": dist,
            "period": period,
        }
        return out

    @property
    def bic(self) -> xr.DataArray:
        """Return the Bayesian Information Criterion.

        BIC = log(n) k - 2 log(L)
        """
        return self._bic(self.dist, self.period)

    def _aic(self, dist, period) -> xr.DataArray:
        sample = self._sample(period)
        dparams = self.fit(dist, period)
        ll = self._ll(dparams, sample)
        out = 2 * len(dparams) - 2 * ll
        out.attrs = {
            "long_name": "Akaike Information Criterion",
            "description": "AIC = 2 k - 2 log(L)",
            "history": update_history(
                "AIC", new_name="aic", parameters=dparams, sample=sample
            ),
            "scipy_dist": dist,
            "period": period,
        }
        return out

    @property
    def aic(self) -> xr.DataArray:
        """Return the Akaike Information Criterion.

        AIC = 2 k - 2 log(L)
        """
        return self._aic(self.dist, self.period)

    @param.depends("period", "metric", watch=True)
    def _update_metrics(self) -> xr.DataArray:
        """Return the metric values for all distributions.

        Parameters
        ----------
        metric : {'aic', 'bic'}
            Information criterion.
        """
        if self.metric not in ["aic", "bic"]:
            raise ValueError(f"Unknown metric {self.metric}")

        # Reset values
        self.metrics = {"aic": None, "bic": None}

        # Update metrics values for given metric
        self.metrics[self.metric] = {
            dist: getattr(self, f"_{self.metric}")(dist, self.period)
            for dist in scipy_dists
        }
        self.param.trigger("metrics")

    @property
    def metrics_da(self) -> xr.DataArray:
        """Return metrics DataArray."""
        if self.metrics[self.metric] is not None:
            vals = [
                val.expand_dims(scipy_dist=[dist])
                for dist, val in self.metrics[self.metric].items()
            ]
            out = xr.concat(vals, dim="scipy_dist")
            out.attrs.pop("scipy_dist")
            return out

    def best_dist(self) -> str:
        """Return the distribution with the best metric value.

        Parameters
        ----------
        metric: {'aic', 'bic'}
          Information criterion that we seek to minimize.
        """
        if self.metrics[self.metric] is None:
            self._update_metrics()

        return self.metrics_da.idxmin("scipy_dist").item()

    def to_dict(self, mode="results"):
        """Return a dict representation of the class."""
        out = super().to_dict(mode=mode)
        if mode == "results":
            out.update(
                {
                    "aic": self.aic.to_dict(),
                    "bic": self.bic.to_dict(),
                    "metrics": self.metrics_da.to_dict(),
                }
            )
        else:
            out[self.metric] = getattr(self, self.metric).item()
        return out

    @property
    def ts_caption(self):
        stations = []
        for vv, sid in self.data.stations.items():
            row = global_config.get_station_meta(sid, VARIABLE_MAPPING.get(vv, vv))
            stations.append(f"{row.station_name} ({sid})")

        if len(stations) == 1:
            st = "station " + stations[0]
        else:
            st = "stations " + ", ".join(stations)

        if self.locale == "fr":
            if len(stations) == 1:
                st = "à la " + st
            else:
                st = "aux " + st

            return f"Série observée de `{self.long_name}` {st}. {self.description}"
        else:
            return f"Observed time series for `{self.long_name}` at {st}. {self.description}"

    @property
    def hist_caption(self):
        period = "–".join(map(str, self.period))
        dist = self.dist
        if self.locale == "fr":
            return (
                f"Densité de probabilité de la distribution {dist}, superposée à l'histogramme de la série "
                f"observée au cours de la période ({period})."
            )
        return (
            f"Probability density function of the {dist} distribution, overlaid on the histogram of the observed "
            f"series during the period ({period})."
        )


class IndicatorSimDA(IndicatorDA):
    # Not used, but required for consistency across classes that need it.
    obs = param.ClassSelector(class_=IndicatorObsDA, doc="Observed indicator")

    model_weights = param.Parameter(None, doc="Weights for the models", allow_refs=True)
    scenario_weights = param.Parameter(None, doc="Weights for the scenarios")
    weights = param.Parameter(None, doc="Combined model and scenario weights")

    def __init__(self, **kwargs):
        # Unstack the data
        kwargs["data"] = (
            kwargs["data"]
            .set_index(realization=("variant_label", "source_id", "experiment_id"))
            .unstack("realization")
        )

        super().__init__(**kwargs)

    def fit(self, dist: str, period: tuple) -> xr.DataArray:
        """Fit the distribution to the data."""
        # Combine time and realizations before statistical fit
        sample = self._sample(period).stack(tr=["time", "variant_label"])
        return xc.indices.stats.fit(sample, dist=dist, dim="tr", method="ML")

    @param.depends("period", watch=True, on_init=True)
    def _update_scenario_weights(self):
        """Compute the scenario weights over the period."""
        if max(self.period) > 2100:
            raise ValueError("Period must end before 2100.")

        w = scen_weights.sel(time=self._slice(self.period)).mean(dim="time")
        # Outside of the SSP period, assume all scenarios equally likely.
        if w.isnull().all():
            w = w.fillna(0.25)
        self.scenario_weights = w

    @param.depends(
        "model_weights", "_update_scenario_weights", watch=True, on_init=True
    )
    def _update_weights(self):
        if self.model_weights is not None:
            self.weights = self.model_weights * self.scenario_weights

    def _dist_method(self, name, arg: xr.DataArray | float):
        mix = XMixtureDistribution(params=self.dparams, weights=self.weights)
        return getattr(mix, name)(arg)

    def experiment_percentiles(self, per) -> xr.Dataset:
        """Return the percentiles computed for each year and experiment.

        Useful for visualizing the distribution of the ensemble.

        Parameters
        ----------
        per : list of int
          The percentiles to compute [0, 100].
        """
        if self.model_weights is None:
            raise ValueError("Please set the model weights.")

        da = self.ts.stack(realization=["source_id", "variant_label"])
        # da.name = self.name

        # Apply weights on the ensemble percentile calculations
        # Here we need to apply weights related to the number of members, as well as model weights
        w = self.model_weights * members(self.ts)
        w = w.expand_dims(variant_label=self.ts.variant_label).stack(
            realization=["source_id", "variant_label"]
        )
        w = w.fillna(0)
        return xclim.ensembles.ensemble_percentiles(da, per, split=True, weights=w)

    def to_dict(self):
        """Return a dict representation of the class."""
        out = super().to_dict()
        out.update(
            {
                "model_weights": self.model_weights.to_dict(),
                "scenario_weights": self.scenario_weights.to_dict(),
                "weights": self.weights.to_dict(),
            }
        )
        return out

    @property
    def ts_caption(self):
        stations = []
        for vv, sid in self.data.stations.items():
            row = global_config.get_station_meta(sid, VARIABLE_MAPPING.get(vv, vv))
            stations.append(f"{row.station_name} ({sid})")

        if len(stations) == 1:
            st = "station " + stations[0]
        else:
            st = "stations " + ", ".join(stations)

        if self.locale == "fr":
            if len(stations) == 1:
                st = "à la " + st
            else:
                st = "aux " + st

            return (
                f"Série projetée de `{self.long_name}` {st}. {self.description} Cliquer sur les items de la légende "
                f"permet de contrôler leur visibilité."
            )
        else:
            return f"Projected time series for `{self.long_name}` at {st}. {self.description} Clicking on legend items allows to control their visibility."

    @property
    def hist_caption(self):
        period = "–".join(map(str, self.period))
        if self.locale == "fr":
            return (
                f"Densité de probabilité combinant les différentes sources d'incertitudes climatiques sur la "
                f"période ({period})."
            )
        return (
            f"Probability density function combining different sources of climate uncertainties over the period "
            f"({period})."
        )


class IndicatorRefDA(IndicatorSimDA):
    obs = param.ClassSelector(class_=IndicatorObsDA, doc="Observed indicator")
    level = param.Number(0.1, allow_refs=True, doc="Significance level for the KS test", allow_None=True)

    _ks = param.Parameter(None, doc="Weights for the models")

    @property
    def period(self):
        return self.obs.period

    @staticmethod
    def ks(obs, ref, level=0.05, rdim="tr") -> xr.DataArray:
        """Kolmogorov-Smirnov test between the observed and simulated data.

        The null hypothesis is that the two distributions are identical.
        If the p-value < 0.05, we reject this hypothesis and return a weight of 0. otherwise we consider the
        distributions similar and return 1.

        Parameters
        ----------
        level : float
            The significance level for the test. Increase the value to include more models. If None, do not perform
            the test and assign a weight of 1 to all models.

        Returns
        -------
        xr.DataArray
          1 if both distributions are similar over the reference period, 0 otherwise.
        """
        from scipy import stats

        if level is None:
            return xr.apply_ufunc(
                lambda x: 1,
                ref,
                input_core_dims=[[rdim]],
                vectorize=True,
                dask="parallelized",
        )

        # We assume that over the reference period, the values are the same for all the experiments.
        def func(r):
            if (~np.isnan(r)).sum() < 10:
                return 0

            return stats.ks_2samp(
                r,
                obs,
                alternative="two-sided",
                method="auto",
                axis=0,
                nan_policy="omit",
                keepdims=False,
            ).pvalue

        p = xr.apply_ufunc(
            func,
            ref,
            input_core_dims=[[rdim]],
            vectorize=True,
            dask="parallelized",
        )
        return p > level

    @param.depends("obs.period", "level", watch=True, on_init=True)
    def _update_ks(self):
        """Find out for which models indicators in the reference period match the observed data."""
        # period of IndicatorRefDA is only a link to the obs one, they are always equal by design
        # Impossible in theory, why does it not update ?
        # if self.obs.period != self.period:
        #     logger.error(f"Reference and observed periods must match. {self.obs.period} <> {self.period}")
        #     return
        #     #raise ValueError("Reference and observed periods must match.")
        # logger.error(f"Reference and observed periods do match.")
        # We can select the first experiment because all experiments have the same values over the historical period.
        ref = self.sample.stack(tr=["time", "variant_label"]).isel(
            experiment_id=0, drop=True
        )
        obs = self.obs.sample

        self._ks = self.ks(obs, ref, self.level, rdim="tr")

    @param.depends("_ks", watch=True)
    def _update_model_weights(self):
        """Compute the weights for the dataset."""
        test = self._ks
        if test is None:
            logger.error(f"{self.xid}: KS is None.")
            return

        # Drop the models that don't match
        ok = test.where(test).dropna("source_id")
        if len(ok) < 2:
            raise ValueError("Not enough models to compute weights.")

        # Compute the weights for valid models
        self.model_weights = model_weights_from_sherwood(
            ok.source_id.values, method="L2Var", lambda_=0.5
        ).sel(source_id=test.source_id)

        logger.info(f"{self.xid}: Model weights computed")


def pievc(sf):
    """Return category from 1 to 7 based on return period.

    Parameters
    ----------
    sf : float
      Survival function (1-CDF) value.
    """
    t = 1 / sf

    bins = [1, 3, 10, 30, 100, 300]
    return np.digitize(t, bins) + 1


class HazardMatrix(BaseParameterized):
    analysis = param.ClassSelector(class_=Analysis, precedence=-1, instantiate=True)
    kind = param.Selector(
        objects={"Probabilities": lambda x: x * 100, "PIEVC": pievc}, instantiate=True
    )

    matrix = param.Dict({}, instantiate=True)

    _levels = {
        "id": ["xid", "descr"],
        "threshold": ["value", "obs_t"],
        "observations": ["obs_sf"],
        "projections": ["ref_sf", "fut_sf", "ratio"],
    }

    _label = {
        "id": {"en": "Identification", "fr": "Identification"},
        "threshold": {"en": "Threshold", "fr": "Seuil"},
        "observations": {"en": "Observations", "fr": "Observations"},
        "projections": {"en": "Projections", "fr": "Projections"},
    }

    _doc = {
        "id": {
            "en": "Identification of hazard and the element it affects",
            "fr": "Identification de l'aléa et de l'élément affecté",
        },
        "threshold": {
            "en": "Climate threshold selection options",
            "fr": "Options de sélection du seuil climatique",
        },
        "observations": {
            "en": "Exceedance probability estimated from observations",
            "fr": "Probabilité de dépassement estimée à partir des observations",
        },
        "projections": {
            "en": "Exceedance probability estimated from model projections",
            "fr": "Probabilité de dépassement estimée à partir des projections de modèles",
        },
    }

    _label = {
        "id": {"en": "Identification", "fr": "Identification"},
        "threshold": {"en": "Threshold", "fr": "Seuil"},
        "observations": {"en": "Observations", "fr": "Observations"},
        "projections": {"en": "Projections", "fr": "Projections"},
    }

    _doc = {
        "id": {
            "en": "Identification of hazard and the element it affects",
            "fr": "Identification de l'aléa et de l'élément affecté",
        },
        "threshold": {
            "en": "Climate threshold selection options",
            "fr": "Options de sélection du seuil climatique",
        },
        "observations": {
            "en": "Exceedance probability estimated from observations",
            "fr": "Probabilité de dépassement estimée à partir des observations",
        },
        "projections": {
            "en": "Exceedance probability estimated from model projections",
            "fr": "Probabilité de dépassement estimée à partir des projections de modèles",
        },
    }

    @param.depends(
        "analysis.obs", "analysis.ref", "analysis.fut", watch=True, on_init=True
    )
    def _update(self):
        """Fill matrix with the indicators from the analysis."""
        if self.analysis is None:
            return

        updated = False
        for key, o in self.analysis.obs.items():
            if key in self.analysis.ref and key in self.analysis.fut:
                if key not in self.matrix:
                    ht = HazardThreshold(
                        obs=o,
                        ref=self.analysis.ref[key],
                        fut=self.analysis.fut[key],
                        hm=self,
                        locale=self.param.locale,
                    )
                    updated = True
                    self.matrix[key] = [ht]

        extras = set(self.matrix.keys()) - set(self.analysis.obs.keys())
        for extra in extras:
            updated = True
            del self.matrix[extra]

        if updated:
            self.param.trigger("matrix")

    def add(self, key: str, index: int):
        """Add a new threshold to the matrix."""
        o = self.analysis.obs[key]
        r = self.analysis.ref[key]
        f = self.analysis.fut[key]

        self.matrix[key].insert(
            index + 1, HazardThreshold(obs=o, ref=r, fut=f, locale=self.param.locale)
        )
        self.param.trigger("matrix")

    def remove(self, key: str, index: int):
        """Remove matrix entry."""
        self.matrix[key].pop(index)
        self.param.trigger("matrix")

    @property
    def df(self) -> pd.DataFrame:
        # Get values
        out = []
        if len(self.matrix) > 0:
            for key, hts in self.matrix.items():
                for ht in hts:
                    out.append(ht.values)

            df = pd.concat(out, axis=1).T  # .set_index(["long_name", "descr"])
            # df = df.set_axis(ht.index.to_flat_index(), axis=1)
            # return df.set_axis(df.columns.to_flat_index(), axis=1).)
            return df

    @property
    def titles(self):
        """Return the labels for the matrix columns."""
        h = {key: self._label[key][self.locale] for key in self._levels.keys()}
        t = list(self.matrix.values())[0][0].titles
        return {**h, **t}

    @property
    def docs(self):
        """Return the description of the matrix columns."""
        h = {key: self._doc[key][self.locale] for key in self._levels.keys()}
        t = list(self.matrix.values())[0][0].docs
        return {**h, **t}

    @property
    def ratios(self) -> pd.Series:
        return pd.Series({key: ht.ratio for key, ht in self.matrix.items()})

    def get_ht(self, index):
        """Get HazardThreshold instance from index."""
        i = 0
        for key, hts in self.matrix.items():
            for ht in hts:
                if i == index:
                    return ht
                i += 1

    def on_click(self, event):
        logger.info(str(event))
        if event.column in ["add", "remove"]:
            i = 0
            for key, hts in self.matrix.items():
                for index, ht in enumerate(hts):
                    if i == event.row:
                        break
                    i += 1
                else:  # Finish without break, skip to next iteration
                    continue
                break

            if event.column == "add":
                self.add(key=key, index=index)
            elif event.column == "remove":
                self.remove(key=key, index=index)

            return True

    def to_dict(self, mode="results"):
        if mode == "results":
            return {
                key: [ht.to_dict() for ht in hts] for key, hts in self.matrix.items()
            }
        if mode == "request":
            uuids = list(sorted(self.matrix.keys()))
            return [
                [ht.to_dict(mode="request") for ht in self.matrix[uuid]]
                for uuid in uuids
            ]

    # TODO: Test locale change
    def from_dict(self, data):
        """Add HazardThresholds from a dict.

        `data` is a mapping from keys for each indicators (UUIDs) to lists of
        dicts for HazardThreshold parameters or full definitions.

        A "parameter" dict only contains T or X and an optional description.
        A "full definition" contains all parameters (see `HazardThreshold._keys`).
        """
        for key, hts in data.items():
            for i, ht in enumerate(hts):
                # If the value is None, this means the user hasn't done anything yet
                # This happens with the dummy entry that is inserted on instantiation of the indicator
                # We remove it as we are adding something meaningful anyway
                if (
                    len(self.matrix[key]) >= (i + 1)
                    and self.matrix[key][i].value is None
                ):
                    self.matrix[key].remove(self.matrix[key][i])
                if "X" in ht or "T" in ht:
                    self.add(key, i)
                    hto = self.matrix[key][i]
                    if "X" in ht and "T" in ht:
                        raise ValueError(
                            "Invalid hazard parameters entry including both X and T."
                        )
                    if "X" in ht:
                        hto.input = "X"
                        hto.value = ht["X"]
                    elif "T" in ht:
                        hto.input = "T"
                        hto.obs_t = ht["T"]
                    else:
                        raise ValueError(
                            "Invalid hazard parameters entry missing X or T."
                        )
                    hto.descr = ht.get("description", f"Aléa #{i:d}")
                else:  # A full definition
                    self.matrix[key].insert(i, HazardThreshold.from_dict(ht))
        self.param.trigger("matrix")


class HazardThreshold(BaseParameterized):
    """Class for hazards thresholds.

    Facilitate going from values to return periods and vice-versa.

    `sf` stands for survival function (1-CDF)
    """

    # Note: not using the specific subclasses because they're not used as parent to the WL classes.
    obs = param.ClassSelector(class_=IndicatorDA, precedence=-1, instantiate=True)
    ref = param.ClassSelector(class_=IndicatorDA, precedence=-1, instantiate=True)
    fut = param.ClassSelector(class_=IndicatorDA, precedence=-1, instantiate=True)

    long_name = param.String(
        label="Name", doc="Indicator name", precedence=-1, instantiate=True
    )
    descr = param.String(label="Élément à risque", doc="Description", instantiate=True)
    xid = param.String(label="ID", doc="xclim indicator call", instantiate=True)

    input = param.Selector(
        objects={"X": "X", "T": "T"},
        doc="Whether the user input is the value or return period.",
        label="Critère",
        instantiate=True,
    )
    value = param.Number(None, doc="Threshold value", label="X", instantiate=True)
    obs_t = param.Number(
        None,
        doc="Return period during the reference period",
        label="T",
        softbounds=(2, 100),
        instantiate=True,
    )

    obs_sf = param.Number(
        None,
        doc="Exceedance probability from observations during the reference period.",
        label="P(Obs > X)",
        instantiate=True,
    )
    ref_sf = param.Number(
        None,
        doc="Exceedance probability from simulations during the reference period.",
        label="P(Ref > X)",
        instantiate=True,
    )
    fut_sf = param.Number(
        None,
        doc="Exceedance probability during the future period.",
        label="P(Fut > X)",
        instantiate=True,
    )
    ratio = param.Number(
        None,
        doc="Ratio of future exceedance probability vs reference",
        label="Fut/Ref - 1",
        instantiate=True,
    )

    hm = param.ClassSelector(
        class_=HazardMatrix, precedence=-1, doc="Matrix of thresholds", instantiate=True
    )

    _keys = [
        # "long_name",
        "xid",
        "descr",
        "obs_t",
        "value",
        "obs_sf",
        "ref_sf",
        "fut_sf",
        "ratio",
    ]

    _label = {
        "input": {"en": "Criteria", "fr": "Critère"},
        "long_name": {"en": "Name", "fr": "Nom"},
        "descr": {"en": "At-risk element", "fr": "Élément à risque"},
        "xid": {"en": "Indicator", "fr": "Indicateur"},
    }

    _doc = {
        "xid": {"en": "Climate hazard", "fr": "Aléa climatique"},
        "descr": {
            "en": "Component affected by hazard",
            "fr": "La composante affectée par l'aléa",
        },
        "obs_t": {
            "en": "Return period during the reference period",
            "fr": "Temps de retour pendant la période de référence",
        },
        "value": {"en": "Climate threshold value", "fr": "Seuil climatique"},
        "obs_sf": {
            "en": "Exceedance probability during the reference period",
            "fr": "Probabilité de dépassement pendant la période de référence",
        },
        "ref_sf": {
            "en": "Exceedance probability during the reference period",
            "fr": "Probabilité de dépassement pendant la période de référence",
        },
        "fut_sf": {
            "en": "Exceedance probability during the future period",
            "fr": "Probabilité de dépassement pendant la période future",
        },
        "ratio": {
            "en": "Ratio of future exceedance probability vs reference",
            "fr": "Ratio de probabilité de dépassement future vs référence",
        },
    }

    @param.depends("obs", watch=True, on_init=True)
    def _update_long_name_and_value_bounds(self):
        """Set default long_name from indicator."""
        if self.obs:
            if self.long_name == "":
                self.long_name = self.obs.long_name
            low = self.obs.isf(1 / self.param.obs_t.softbounds[0]).item()
            high = self.obs.isf(1 / self.param.obs_t.softbounds[1]).item()
            self.param.value.softbounds = (low, high)

    @param.depends("obs", watch=True, on_init=True)
    def _update_xid(self):
        """Set default long_name from indicator."""
        if self.xid == "" and self.obs:
            self.xid = self.obs.xid

    @param.depends("value", "obs.dparams", watch=True, on_init=True)
    def _update_from_value(self):
        """Compute return period from value."""
        if self.value is not None and self.input == "X":
            self.obs_sf = self.obs.sf(self.value).item()
            if self.obs_sf == 0:
                self.obs_t = np.inf
            else:
                self.obs_t = 1 / self.obs_sf

    @param.depends("obs_t", "obs.dparams", watch=True, on_init=True)
    def _update_from_obs_t(self):
        """Compute value from return period."""
        if self.obs_t is not None and self.input == "T":
            self.obs_sf = 1 / self.obs_t
            self.value = self.obs.isf(self.obs_sf).item()

    @param.depends("value", "ref.dparams", "fut.dparams", watch=True, on_init=True)
    def _update_sim(self):
        """Compute simulated exceedance probabilities and ratio."""
        if self.value is not None:
            self.ref_sf = self.ref.sf(self.value).item()
            self.fut_sf = self.fut.sf(self.value).item()
            if self.ref_sf == 0:
                self.ratio = np.nan
            else:
                self.ratio = self.fut_sf / self.ref_sf - 1

    @property
    def values(self) -> pd.Series:
        """Return values Series."""
        values = [getattr(self, key, None) for key in self._keys]
        return pd.Series(values, index=self._keys)

    def to_dict(self, mode="results"):
        if mode == "results":
            return dict(zip(self._keys, self.values))
        if mode == "request":
            out = {}
            if self.descr:
                out["description"] = self.descr
            if self.input == "T":
                out["T"] = self.obs_t
            elif self.input == "X":
                out["X"] = self.value
            return out

    @property
    def titles(self) -> dict:
        """Return labels Series."""
        return {key: getattr(self.param[key], "label") for key in self._keys}

    @property
    def docs(self):
        return {key: getattr(self.param[key], "doc") for key in self._keys}

    def on_edit(self, event):
        """Trigger the edit event."""
        logger.info(f"{event.column}: {event.value}")

        name = event.column
        if name == "value":
            self.input = "X"
        elif name == "obs_t":
            self.input = "T"

        setattr(self, name, event.value)


def haversine(lon1: float, lat1: float, lon2: np.array, lat2: np.array):
    """Calculate the great circle distance between points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    return 6378.137 * c  # km


def xid_from_da(da: xr.DataArray) -> str:
    """Return the xclim indicator id from a DataArray."""
    from xclim.core.formatting import gen_call_string

    return gen_call_string(da.id, **da.params)
