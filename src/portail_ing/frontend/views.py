"""
# Panel views

This module contains the views for the frontend of the application. The views implemented are:
  - StationViewer: Display a selector and map for the stations available per variable;
  - IndicatorsViewer: Show list of indicators available for computation and buttons to select them;
  - IndicatorBaseViewer: The panel for an individual Indicator and its parameters.

TODO: Translate
 - button names

Use layout property to inspect components
"""

import datetime as dt
import io
import json
import os
import textwrap
import time
import traceback
import zipfile
from functools import partial
from pathlib import Path

import geopandas as gpd
import holoviews as hv
import hvplot.pandas  # noqa: F401
import hvplot.xarray  # noqa: F401
import ipyleaflet
import numpy as np
import pandas as pd
import panel as pn
import param
import requests
import xarray as xr
import yaml
from bokeh.models.widgets.tables import DateFormatter, NumberFormatter
from ipyleaflet import (
    AwesomeIcon,
    Circle,
    CircleMarker,
    Map,
    Marker,
    WidgetControl,
    basemaps,
)
from ipywidgets import HTML, Button, FloatText, HBox, IntSlider, Layout, VBox
from panel.viewable import Viewable

from portail_ing.common import config as global_config
from portail_ing.common.logger import get_logger

from . import parameters as p

logger = get_logger("app")

hv.extension("bokeh")
LOCAL_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = Path(os.environ.get("FRONTEND_CONFIG_DIR", LOCAL_DIR / "config"))

colors = {
    "ssp119": "#1e9684",
    "ssp126": "#1d3354",
    "ssp245": "#eadd3d",
    "ssp370": "#f21111",
    "ssp585": "#840b22",
    "historical": "#7E7F7F",
}

labels = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}


class LocalizedFileDownload(pn.widgets.FileDownload, p.BaseParameterized):
    # Secret panel trick to remove those params from the Bokeh model
    _rename = {"locale": None, "_value": None}


class Text(p.BaseParameterized):
    download = param.String("Download")
    load_json = param.String("Load JSON")
    fact_sheets = param.String("Fact Sheets")

    _value = {
        "download": {"en": "## Downloads", "fr": "## Téléchargements"},
        "load_json": {
            "en": "## Import JSON request",
            "fr": "## Importer une requête JSON",
        },
    }


class Viewer(pn.viewable.Viewer):
    _conf = {}

    def __init__(self, config=None, **params):
        super().__init__(**params)
        self._conf = config


class StationViewer(Viewer):
    """Display a selector and map for the stations available."""

    # The station parameter
    station = param.ClassSelector(
        class_=p.Station,
        allow_refs=True,
        nested_refs=True,
        instantiate=False,
        per_instance=False,
    )
    site = param.ClassSelector(
        class_=p.Site,
        allow_refs=True,
        nested_refs=True,
        instantiate=False,
        per_instance=False,
    )
    map_param = param.ClassSelector(
        class_=p.Map,
        allow_refs=True,
        nested_refs=True,
        instantiate=False,
        per_instance=False,
    )

    #
    table = param.Dict({})
    # Selected marker color
    _selected_marker_color = "black"

    _legend_title = {"fr": "Variables", "en": "Variables"}

    _label = {""}

    site_circle = {}
    site_widgets = {}

    last_seen_zoom = np.nan
    updating_paramwidget = {}

    def __init__(self, **params):
        """__init__(config, station, state)"""
        self.map = None
        super().__init__(**params)
        if self.station is None:
            raise ValueError("StationViewer must be instantiated with a Station class")
        if self.site is None:
            raise ValueError("StationViewer must be instantiated with a Site class")
        if self.map_param is None:
            raise ValueError("StationViewer must be instantiated with a Map class")
        # self._conf = config
        # Station markers
        self.variables = self.station.variables
        self.circle_marker_groups = {}
        self.icon_marker_groups = {}
        self.site_widgets = {}
        self.site_circle = {}

        self.map_container = None
        self.table_container = None

        self.layout = self._default_layout()

    def get_modified_circle_radius(self, radius, center):
        """
        Given a radius in km, get a modified radius in meters that is
        valid at the equator (i.e. Leaflet's radius)
        """
        x, y = p.latlng_to_proj(center[0], center[1])

        # find the radius in latlng space:
        radians = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        circle_eastings = x + radius * 1000 * np.cos(radians)
        circle_northings = y + radius * 1000 * np.sin(radians)
        xs, ys = p.proj_to_latlng(circle_eastings, circle_northings)
        rad_lat = np.max(ys - center[0])
        rad_lon = np.max(xs - center[1])

        # find the radius in meters:
        _, radius_m = p.latlng_to_proj(rad_lat, rad_lon)
        # print("get_modified_circle_points", radius, center, radius_m)
        return int(radius_m)

    def init_site_val(self):
        update = False
        if self.site.lat and self.site.lon:
            center = (self.site.lat, self.site.lon)
        else:
            center = (self._conf["site"]["lat"], self._conf["site"]["lon"])
            update = True

        if self.site.radius:
            radius = self.site.radius
        else:
            radius = self._conf["site"]["radius"]
            update = True
        if update:
            logger.debug(
                f"Update self.site ({self.site.lat}, {self.site.lon}, {self.site.radius}), to {center}, {radius}"
            )
            self.site.param.update(lat=center[0], lon=center[1], radius=radius)
        else:
            logger.debug(
                f"using self.site ({self.site.lat}, {self.site.lon}, {self.site.radius}) for center, radius"
            )
        return center, radius, self.site.enabled

    def init_map_loc(self):
        update = False
        if self.map_param.clon and self.map_param.clat:
            center = (self.map_param.clat, self.map_param.clon)
        else:
            center = self._conf["map"]["center"]
            update = True

        if self.map_param.z:
            zoom = self.map_param.z
        else:
            zoom = self._conf["map"]["zoom"]
            update = True

        if update:
            self.map_param.param.update(clat=center[0], clon=center[1], z=zoom)

        return center, zoom

    def init_site_circle(self):
        # find initial parameters from either self.site, if avail, or _conf
        center, radius, enabled = self.init_site_val()
        logger.debug(f"StationViewer.init_site_circle: {center} {radius} {enabled}")

        # dot at center of circle:
        center_marker = CircleMarker(
            location=center,
            name="selection_centre",
            radius=0,
            weight=3,
            color="blue",
            visible=enabled,
            draggable=True,
        )

        # border/fill of the circle:
        border_marker = Circle(
            location=center,
            name="selection_circle",
            radius=self.get_modified_circle_radius(radius, center),
            weight=3,
            color="blue",
            fill_opacity=0.1,
        )

        self.map.add(border_marker)
        # self.map.add(center_marker)
        self.site_circle = {
            "border_marker": border_marker,
            "center_marker": center_marker,
        }

    def init_site_widgets(self):
        center, radius, enabled = self.init_site_val()
        logger.debug(f"StationViewer.init_site_widgets: {center} {radius} {enabled}")

        # widgets to control the site parameters:
        site_toggle = Button(
            icon="dot-circle-o",
            layout=Layout(width="12%", height="90%"),
            disabled=False,
        )

        lat_editor = FloatText(
            placeholder="Lat", value=center[0], layout=Layout(width="40%"), step=0.1
        )
        lon_editor = FloatText(
            placeholder="Lon", value=center[1], layout=Layout(width="40%"), step=0.1
        )

        radius_slider = IntSlider(
            description="Distance (km):",
            min=0,
            max=250,
            value=radius,
            continuous_update=False,
            layout=Layout(width="90%"),
            style={"description_width": "initial"},
        )

        # dict of widgets:
        self.site_widgets = {
            "site_toggle": site_toggle,
            "lat_editor": lat_editor,
            "lon_editor": lon_editor,
            "radius_slider": radius_slider,
        }

        # state observers:
        site_toggle.on_click(self.site_toggle_changed)
        lat_editor.observe(self.site_lat_editor_changed, "value")
        lon_editor.observe(self.site_lon_editor_changed, "value")
        radius_slider.observe(self.site_radius_slider_changed, "value")

        widgetbox = VBox(
            [HBox([site_toggle, lat_editor, lon_editor], width="200px"), radius_slider]
        )
        widgetcontrol = WidgetControl(
            widget=widgetbox, position="topright", max_width=300
        )
        self.map.add(widgetcontrol)

    def update_ipywidget_from_param(self, subparam_name, widget, **traitvalues):
        if not widget:
            return
        if not self.updating_paramwidget.get(subparam_name):
            self.updating_paramwidget[subparam_name] = True
            with widget.hold_trait_notifications():
                for trait, value in traitvalues.items():
                    setattr(widget, trait, value)
            self.updating_paramwidget[subparam_name] = False

    def update_param_from_ipywidget(self, param_name, **kwargs):
        param_vals = self.param.values()
        if param_name in param_vals and isinstance(
            param_vals[param_name], param.Parameterized
        ):
            update = {}
            for key, val in kwargs.items():
                if not self.updating_paramwidget.get(key):
                    self.updating_paramwidget[key] = True
                    update[key] = val
            param_vals[param_name].param.update(**update)
            for key in update:
                self.updating_paramwidget[key] = False
        else:
            raise ValueError(f"Could not find param.Parameterized {param_name}")

    def site_toggle_changed(self, issuer):
        """Toggle circle visibility."""
        self.site.enabled = not self.site.enabled

    def site_lat_editor_changed(self, event):
        """Change lat via the lat_editor widget."""
        self.update_param_from_ipywidget("site", lat=round(event["new"], 4))

    def site_lon_editor_changed(self, event):
        """Change lat via the lat_editor widget."""
        self.update_param_from_ipywidget("site", lon=round(event["new"], 4))

    def site_radius_slider_changed(self, event):
        """Change lat via the lat_editor widget."""
        self.update_param_from_ipywidget("site", radius=event["new"])

    @param.depends("site.lat", "site.lon", "site.radius", watch=True)
    def site_on_change(self, *args, **kwargs):
        params = self.site.param.values()
        enabled = enabled = params.get("enabled")
        if not enabled:
            return

        radius = self.site.radius
        lat, lon = self.site.lat, self.site.lon
        center = (lat, lon)
        mod_radius = self.get_modified_circle_radius(radius, center)

        self.update_ipywidget_from_param(
            "radius", self.site_widgets.get("radius_slider"), value=radius
        )
        self.update_ipywidget_from_param(
            "lat", self.site_widgets.get("lat_editor"), value=lat
        )
        self.update_ipywidget_from_param(
            "lon", self.site_widgets.get("lon_editor"), value=lon
        )

        self.update_ipywidget_from_param(
            "border_marker",
            self.site_circle.get("border_marker"),
            radius=mod_radius,
            location=center,
        )
        self.update_ipywidget_from_param(
            "center_marker", self.site_circle.get("center_marker"), location=center
        )

    def map_bounds_link(self, *events):
        with param.parameterized.batch_call_watchers(self.map_param):

            for event in events:
                if event["name"] == "center":
                    # setting clat, clon to 4 decimals is enough accuracy to prevent jitter.
                    self.update_param_from_ipywidget(
                        "map_param",
                        clat=round(event["new"][0], 4),
                        clon=round(event["new"][1], 4),
                        z=self.map.zoom,
                    )
                # this causes jitter, since the center updates whenever the zoom changes.
                # if event['name'] == 'zoom':
                #    logger.info(f'map_bounds_link zoom {event["new"]}')
                #    self.update_param_from_ipywidget('map_param', z=event['new'])

    @param.depends("map_param.clat", "map_param.clon", "map_param.z", watch=True)
    def map_bounds_change(self):
        if self.map is None:
            return
        # do not update self.map, it will cause jitter due to its independent asynchronous parameters:
        #  self.update_ipywidget_from_param(
        #      "map_param",
        #      self.map,
        #      zoom=self.map_param.z,
        #      center=[self.map_param.clat, self.map_param.clon],
        #  )
        self.update_visible_markers()
        self.last_seen_zoom = self.map_param.z

    @param.depends("site.enabled", watch=True)
    def site_on_toggle(self):
        params = self.site.param.values()
        enabled = params.get("enabled")
        border_marker = self.site_circle.get("border_marker")
        center_marker = self.site_circle.get("center_marker")
        if (not border_marker) or (not center_marker):
            return
        if enabled:
            border_marker.weight = 3
            border_marker.fill_opacity = 0.1
            center_marker.weight = 3
        else:
            border_marker.weight = 0
            border_marker.fill_opacity = 0
            center_marker.weight = 0

    def target_marker_widget_control(self):
        """Create a widget to display the coordinates of the target site, and a marker to drag."""
        self.init_site_circle()
        self.init_site_widgets()

    @param.depends("station.site_df", watch=True)
    def update_tabulator_tables_from_site(self):
        if not self.table:
            return
        for var, table in self.station.site_df.items():
            station_id = self.station.station_id[var]
            self.table[var].value = table[self.table[var].value.columns].reset_index(
                drop=True
            )
            if station_id is not None:
                sel = table[table["station"] == station_id]
                self.table[var].selection = sel.index.to_list()
                logger.info(
                    f"update_tabulator_tables_from_site {var} {station_id} {sel.index.to_list()}"
                )

    @param.depends("station.site_df", watch=True)
    def update_site_markers(self):
        if self.map is None:
            return
        for var, table in self.station.site_df.items():
            station_id = self.station.station_id[var]
            layergroup = self.icon_marker_groups.get(var)
            if layergroup is None:
                continue
            layergroup.clear()
            for ind, row in table.iterrows():
                icon_opts = self._conf["icon"][var].copy()
                zind = 0
                if row.station == station_id:
                    icon_opts["icon_color"] = self._selected_marker_color
                    zind = 200  # offset for hover = 250
                # zIndex = pos.y + zIndexOffset (leaflet)
                marker = Marker(
                    location=(row.lat, row.lon),
                    title=var + " - " + row.station_name,
                    draggable=False,
                    rotation_angle=self._conf["icon"][var].get("angle", 0),
                    rotation_origin="bottom center",
                    z_index_offset=zind,
                    rise_on_hover=True,
                    rise_offset=1000,
                    icon=AwesomeIcon(**icon_opts),
                )
                marker.on_click(self.create_marker_callback(var, row.station))
                layergroup.add(marker)

    def select_station(self, variable, station, origin):
        """Select a station.

        If station was already selected, deselect it by setting back the station to None.

        Parameters
        ----------
        variable : str
            The variable for which the station is selected.
        station : str
            The station ID to select.
        origin : {"map", "table"}
            Where the callback comes from.

        Notes
        -----
        When we set `station.<variable>`, it triggers `site_df`. In turn, this updates both the markers
        (`update_site_markers`) and the tables (`update_station_tables_from_site`), so there is no
        need to do this explicitly here.
        """
        logger.info(f"Selected station from {origin}: {variable} {station}")
        self.load()
        # If the same station is clicked, deselect it.
        if self.station.station_id[variable] == station:
            station = None
        setattr(self.station, variable, station)

    def load(self):
        self.map_container.loading = True
        self.map_container.loading = True

    @param.depends("station.selected_df", watch=True)
    def unload(self):
        self.map_container.loading = False
        self.table_container.loading = False

    def create_marker_callback(self, variable, station):
        """Create a callback to select a station from the map."""

        def callback(*args, **kwargs):
            self.select_station(variable, station, origin="map")

        return callback

    def make_table_click_callback(self, var):
        """Create custom function for each table.

        We need this to pass information to the callback function, since `on_click` does not accept
        arguments other than the callback function itself.
        """

        def callback(event, **kwargs):
            """Set the variable to the station id."""
            if event.row is None:
                station = None
            else:
                i = event.row
                # Not sure why the only way it works is through a private prop...
                # IIUC, there might be a fix upcoming in panel 1.4.6
                station = self.table[var].value.iloc[i].station
                # pn.state.notifications.info(f"{self.table[var].selection}, {i}, {station}, {self.table[var]._processed.iloc[i].station} {self.table[var].value.iloc[i].station}")
            self.select_station(var, station, origin="table")

        return callback

    # @debounce(0.5)
    def get_bounds_centre(self):
        geom = self.map.bounds
        if not geom:
            geom = self._conf["initial_bounds"]
        (south, west), (north, east) = geom
        centre = ((east + west) / 2, (north + south) / 2)
        return north, east, south, west, centre

    def update_visible_markers(self):
        """On map move, update marker locations"""
        # initial bounds:
        north, east, south, west, centre = self.get_bounds_centre()
        if not (
            self.map_param.clat is None
            or self.map_param.clat == 0
            or self.map_param.clon is None
            or self.map_param.clon == 0
        ):
            centre = (self.map_param.clon, self.map_param.clat)

        ne_easting, ne_northing = p.latlng_to_proj(north, east)
        sw_easting, sw_northing = p.latlng_to_proj(south, west)

        c_easting, c_northing = p.latlng_to_proj(centre[1], centre[0])
        radius = max(abs(ne_easting - sw_easting), abs(ne_northing - sw_northing)) / 2
        if self.map_param.z < self.last_seen_zoom:
            radius *= 2
        elif self.map_param.z > self.last_seen_zoom:
            radius /= 2
        # stations within:
        for var in self.station.variables:
            var_df = self.station.get_within(
                var,
                distance=radius,
                easting=c_easting,
                northing=c_northing,
                return_sorted=False,
            )

            # var_df['geometry'] = shapely.Point(var_df.easting, var_df.northing)
            var_df = gpd.GeoDataFrame(
                var_df, geometry=gpd.points_from_xy(var_df.lon, var_df.lat), crs=4326
            )
            markers = ipyleaflet.GeoData(
                geo_dataframe=var_df,
                point_style={
                    "radius": 0,
                    "color": self._conf["icon"][var]["button_color"],
                    "opacity": 0.5,
                    "weight": 5,
                },
            )
            self.circle_marker_groups[var].clear()
            self.circle_marker_groups[var].add(markers)
        return

    def interact_map(self, event, coordinates=None, type=None, **kwargs):
        if event != "interaction" or type != "click" or coordinates is None:
            return
        lat = round(coordinates[0], 3)
        lon = round(coordinates[1], 3)
        if self.site.enabled:
            self.site.param.update(lat=lat, lon=lon)
        return

    def map_view(self):
        """Draw map with station markers, one per variable."""
        # Basemap
        center, zoom = self.init_map_loc()
        self.map = Map(
            basemap=basemaps.CartoDB.Positron,
            zoom=zoom,
            center=center,
            layout=Layout(**self._conf["map_layout"]),
            scroll_wheel_zoom=True,
        )

        # Add target marker and widget control
        self.target_marker_widget_control()

        self.map.observe(self.map_bounds_link, names=["zoom", "center"])
        # Add and store markers for each variable and each station.
        # station_markers  []

        for var in self.station.variables:
            self.circle_marker_groups[var] = ipyleaflet.LayerGroup(layers=[])
            self.icon_marker_groups[var] = ipyleaflet.LayerGroup(layers=[])
            self.map.add(self.circle_marker_groups[var])
            self.map.add(self.icon_marker_groups[var])

        self.map.on_interaction(self.interact_map)
        # Store legend so we can modify its widget later (for translation)
        self.legend = WidgetControl(
            widget=self.make_legend(),
            position="bottomright",
            max_width=300,
            transparent_bg=False,
        )

        self.map.add(self.legend)
        return self.map

    def update_markers_and_unlisten(self, *args):
        if (self.map.bounds is None) or (len(self.map.bounds) == 0):
            # keep listening until map.bounds is ready.
            return
        self.last_seen_zoom = self.map.zoom
        self.update_visible_markers()
        self.station.update_tables()
        # listener to site_df, updated above.
        # self.update_site_markers()
        self.map.unobserve(self.update_markers_and_unlisten, "bounds")

    def setup_stations(self):
        self.map.observe(self.update_markers_and_unlisten, "bounds")
        # if bounds have been setup before setup_stations is called, the observe is never called on the first load.
        if self.map.bounds is not None and len(self.map.bounds) > 0:
            self.update_markers_and_unlisten()

    def toggle_variable_callback(self, button, variable):
        """Callback to toggle visibility of a variable."""
        ind = [
            ind
            for ind, var in enumerate(self.station.variables.keys())
            if var == variable
        ][0]

        def toggle_variable(event):
            if variable in self.variables:
                self.variables = [var for var in self.variables if var != variable]
                if self.map and self.circle_marker_groups.get(variable):
                    self.map.remove_layer(self.circle_marker_groups[variable])
                if self.map and self.icon_marker_groups.get(variable):
                    self.map.remove_layer(self.icon_marker_groups[variable])
                if self.table_container.object:
                    self.table_container.object[ind].visible = False
                button.style.button_color = "#d3d3d3"
                button.style.text_color = "black"
            else:
                self.variables.append(variable)
                if self.map and self.circle_marker_groups.get(variable):
                    self.map.add(self.circle_marker_groups[variable])
                if self.map and self.icon_marker_groups.get(variable):
                    self.map.add(self.icon_marker_groups[variable])
                if self.table_container.object:
                    self.table_container.object[ind].visible = True
                button.style.button_color = self._conf["icon"][variable]["button_color"]
                button.style.text_color = self._conf["icon"][variable]["icon_color"]

        return toggle_variable

    def make_legend(self):
        """Return map legend javascript code snippet."""
        items = [HTML(f"<b>{self._legend_title[self.station.locale]}</b>")]
        for var in self.station.variables:
            var_name = self.station.label(var)
            icon = self._conf["icon"][var]["name"]
            # text = f"<i class="fa fa-li fa-{icon}"></i>{var_name}"
            button = Button(
                description=var_name,
                icon=icon,
                button_style="info",
            )
            button.style.button_color = self._conf["icon"][var]["button_color"]
            button.style.text_color = self._conf["icon"][var]["icon_color"]
            button.on_click(self.toggle_variable_callback(button, var))
            items.append(button)

        return VBox(items)

    def create_tables(self):
        """

        TODO: Set frozen_rows for selected rows ?
        selectable_rows
        """
        df = pd.DataFrame(
            columns=self._conf["table"]["columns"],
        )
        labels = {
            key: self.station._label[key][self.station.locale] for key in df.columns
        }
        filters = {
            "station": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter station ID",
            },
            "station_name": {
                "type": "input",
                "func": "like",
                "placeholder": "Enter title",
            },
            "variable": {"type": "list", "placeholder": "Select variable"},
            "valid_months": {
                "type": "number",
                "func": ">=",
                "placeholder": "Min months",
            },
            # "distance": {"type": "number", "func": "<=", "placeholder": "Max distance"},
        }

        # Split station metadata into a distinct tables, one for each core variable.

        for v in self.station.variables:
            vdf = self.station.site_df.get(v, df.copy())

            # TODO: When recreating the table (like when switching locale), we lose the selection
            # TODO: Sorting screws up the selection.
            sorters = [{"field": "distance", "dir": "asc"}]
            self.table[v] = pn.widgets.Tabulator(
                vdf[self._conf["table"]["columns"]],
                disabled=True,
                pagination="remote",
                page_size=10,
                page=1,
                sizing_mode="stretch_width",
                hierarchical=False,
                selectable="toggle",
                show_index=False,
                titles=labels,
                formatters={
                    "station": "",
                    "distance": NumberFormatter(format="0.0"),
                    "start": DateFormatter(format="%Y-%m"),
                    "end": DateFormatter(format="%Y-%m"),
                },
                header_filters=filters,
                sorters=sorters,
                theme="bootstrap4",
            )

            self.table[v].on_click(self.make_table_click_callback(v))

    def table_view(self):
        self.create_tables()

        col = []
        for v, table in self.table.items():
            name = self.station._label[v][self.station.locale]
            col.append(pn.Column(pn.pane.Markdown(f"## {name}"), table))

        return pn.Column(*col)

    def export_map(self, path="station_map.png"):
        """Save map to path."""
        # Untested
        from ipywebrtc import ImageRecorder, WidgetStream

        widget_stream = WidgetStream(widget=self.map, max_fps=1)
        image_recorder = ImageRecorder(stream=widget_stream)
        with open(path, "wb") as f:
            f.write(image_recorder.image.value)

    @param.depends("station.locale", watch=True)
    def _translate(self):
        logger.debug(f"Updating StationViewer with locale {self.station.locale}.")
        self.map_container.loading = True
        self.table_container.loading = True
        if hasattr(self, "legend"):
            self.legend.widget = self.make_legend()
        self.table_container.update(self.table_view())
        self.map_container.loading = False
        self.table_container.loading = False

    def _dynamic_load(self):
        """Update the layout of the Row object to include the map."""
        logger.debug("StationViewer._dynamic_load.")
        self.map_view()
        # Set up the map
        self.map_container.update(self.map)
        # need to reset the table_view to refer to caller, instead of self (?)
        self.table_container.update(self.table_view())
        self.setup_stations()
        self.map_container.loading = False
        self.table_container.loading = False

    def _dynamic_unload(self):

        logger.debug("StationViewer._dynamic_unload")
        self.map_container = None

        self.circle_marker_groups = {}
        self.icon_marker_groups = {}
        self.table = {}

        self.site_widgets = {}
        self.site_circle = {}

        self.variables = self.station.variables

        if self.map is not None:
            self.map.clear()
        self.map = None
        self.table_container = None
        self.last_seen_zoom = np.nan

        default_layout = self._default_layout()
        self.layout.objects = default_layout.objects

    def _default_layout(self):
        if self.map_container is None:
            self.map_container = pn.pane.Placeholder(
                loading=True, **self._conf["pre_map_layout"]
            )
        if self.table_container is None:
            self.table_container = pn.pane.Placeholder(
                loading=True, **self._conf["pre_map_layout"]
            )

        return pn.Column(self.map_container, self.table_container)

    def __panel__(self):
        """Return panel."""
        # The left column shows the class' "public" parameters, keyed by variable name.
        # These are defined at run time in parameters.Station.__init__.
        # leg = pn.Row(pn.Spacer(sizing_mode="stretch_width"), self.legend())
        # tv = pn.Column(pn.panel(self.station), leg)  # , widgets={"tas": {"type":

        # pn.widgets.AutocompleteInput,
        # "width": 300},
        # "pr": {"type": pn.widgets.AutocompleteInput, "width": 300}}))

        logger.debug("StationViewer.__panel__ Loading panel object")

        return self.layout


class IndicatorComputationViewer(pn.viewable.Viewer):
    """The panel for an individual Indicator and its parameters."""

    ind = param.ClassSelector(class_=p.IndicatorComputation)

    def __panel__(self):
        """Return panel view."""
        header = pn.pane.Markdown(f"### {self.ind.base.title}")
        abstract = pn.pane.Markdown(f"_{self.ind.base.abstract}_")

        mw = 80
        widgets = {}
        for key in self.ind.args.param.values():
            p = getattr(self.ind.args.param, key)
            if (p.precedence or 1) < 0:
                continue
            if key in ["start_m", "end_m"]:
                widgets[key] = {"width": mw, "disabled": self.ind.param.computed}
            elif hasattr(p, "objects"):
                widgets[key] = {
                    "width": mw,
                    "widget_type": pn.widgets.Select,
                    "size": 1,
                    "disabled": self.ind.param.computed,
                }
            else:
                widgets[key] = {
                    "type": pn.widgets.FloatInput,
                    "width": 150,
                    "step": 1,
                    "format": "0.0",
                    "disabled": self.ind.param.computed,
                    # "format": PrintfTickFormatter(format=f"%.1f {default_units}")
                }
        widgets["variables"] = {"visible": False}

        params = pn.Param(
            self.ind.args,
            default_layout=pn.Row,
            widgets=widgets,
            show_name=False,
        )

        return pn.Column(
            header, abstract, params, width_policy="max", sizing_mode="stretch_width"
        )


class IndicatorListViewer(Viewer):
    """Show indicators' selection panel, and the computation parameters for each selected indicator."""

    inds = param.ClassSelector(class_=p.IndicatorList)

    _conf = param.Dict()

    @param.depends("inds.locale", "inds.station_id")
    def indicator_list_view(self):
        """Return a panel with the indicators available and a button to add them to the selection."""
        top_rows = []
        for cname, conf in self._conf.items():
            rows = []
            for iid in conf["items"]:

                ind = self.inds.indicators[iid]

                # Indicator title
                title = pn.pane.Markdown(
                    ind.title, min_width=300, sizing_mode="stretch_width"
                )

                # Shows which variables have data
                variables = pn.Param(
                    ind.param.variables,
                    widgets={
                        "variables": {
                            "type": pn.widgets.CheckButtonGroup,
                            "disabled": True,
                            "stylesheets": [checkmark],
                            "width": 120,
                        }
                    },
                    width=120,
                )

                # Button to add indicator
                button = pn.widgets.Button(
                    name="＋",
                    width=30,
                    button_type="primary",
                    disabled=ind.has_data is False,
                )
                button.on_click(partial(self.add_indicator_action, iid=iid))

                rows.append(pn.Row(title, variables, pn.HSpacer(max_width=10), button))

            card = pn.Card(*rows, title=conf["desc"][self.inds.locale], collapsed=True)
            top_rows.append(card)

        return pn.Column(*top_rows)

    def add_indicator_action(self, event, iid):
        """Callback when an indicator is selected."""
        # Modifies `selected`, which triggers `selected_view` to update.
        try:
            self.inds.add(iid)
        except OverflowError:
            pn.state.notifications.error(
                "Le nombre maximal d'indicateur est atteint. Veuillez en retirer avant d'en ajouter de nouveaux."
            )

    @param.depends("inds.selected", "inds.locale")
    def selected_view(self):
        """The list of indicators selected by the user."""
        rows = []
        for uuid, ind in self.inds.selected.items():
            icv = IndicatorComputationViewer(ind=ind)

            button = pn.widgets.ButtonIcon(icon="square-x", size="2em")
            button.on_click(partial(self.remove_indicator_action, uuid=uuid))

            pbars = [
                pn.indicators.Progress(
                    value=ind.obs_job.param.progress,
                    active=ind.obs_job.param.active,
                    width=60,
                ),
                pn.indicators.Progress(
                    value=ind.sim_job.param.progress,
                    active=ind.sim_job.param.active,
                    width=60,
                ),
            ]

            indv = pn.Row(icv, pn.Column(button, *pbars, width=100))
            rows.append(indv)
        return pn.Column(*rows)

    def remove_indicator_action(self, event, uuid):
        """Callback when an indicator is removed from the list of selected indicators."""
        # Modifies `selected`, which triggers `selected_view` to update.
        self.inds.remove(uuid)

    def compute_action(self, event):
        """Callback when the user clicks the compute button."""
        if (msg := self.inds.ping_backend()) is not True:
            pn.state.notifications.info(msg)
            return

        try:
            self.inds.post_all_requests()
        except requests.exceptions.RequestException as e:
            pn.state.notifications.info(
                f"Erreur lors de l'envoi des requêtes de calcul: {e}"
            )

        logger.debug(
            "IndicatorListViewer.compute_action(): Launching monitoring callback"
        )
        # The timeout is to short, future computations take 2 min on my laptop.
        self._monitor = pn.state.add_periodic_callback(
            self.monitor_jobs, period=1500, start=True, timeout=20000
        )

    def monitor_jobs(self):
        if self.inds.monitor_jobs():
            self._monitor.stop()

    @param.depends("inds.locale")
    def select(self):
        return pn.Column(
            pn.pane.Markdown(f"## {self.inds.param.indicators.label}"),
            self.indicator_list_view,
        )

    @param.depends("inds.locale")
    def selected(self):
        return pn.Column(
            pn.pane.Markdown(f"## {self.inds.param.selected.label}"),
            self.selected_view,
            self.launch_view,
        )

    @param.depends("inds.locale")
    def launch_view(self):
        # Compute button
        name = {"fr": "Calculer les indicateurs", "en": "Compute indicators"}
        self.button = pn.widgets.Button(
            name=name[self.inds.locale],
            width=200,
            button_type="primary",
            disabled=self.inds.param.allcomputed,
            icon="automation",
        )

        # Manual status update
        # self.status_update = pn.widgets.Button(
        #     name="Rafraichir",
        #     width=200,
        #     button_type="success",
        #     disabled=False,
        # )

        # Add callbacks to the buttons
        self.button.on_click(self.compute_action)
        # self.status_update.on_click(self.inds.monitor_jobs)

        # The view of the compute button and any feedback from the backend ?
        return pn.Row(self.button)

    def __panel__(self):
        """Return panel."""
        return pn.Row(self.select, pn.HSpacer(max_width=10), self.selected)


class AbstractAnalysisViewer(Viewer):
    """View to select reference period and statistical distributions for indicators."""

    analysis = param.ClassSelector(class_=p.Analysis)
    results = param.Dict()
    tab = param.String(allow_refs=True, precedence=-1)
    # view = param.ClassSelector(class_=pn.pane.Placeholder)

    def __init__(self, **params):
        super().__init__(**params)
        self.view = pn.pane.Placeholder()
        self.results = {}
        self.results_view()

    config = param.Dict({})
    empty_title = param.String("Pas de données disponibles")
    empty_message = param.String(
        "Lancez les calculs d'indicateurs avant d'afficher les résultats."
    )

    _label = {
        "empty_title": {"fr": "Pas de données disponibles", "en": "No data available"},
        "empty_message": {
            "fr": "Lancez les calculs d'indicateurs avant d'afficher les résultats.",
            "en": "Run the indicator calculations before displaying the results.",
        },
    }

    @param.depends("analysis.locale")
    def empty_view(self):
        """Message displayed when results are not yet computed."""
        return pn.pane.Alert(
            f"### {self.empty_title}\n{self.empty_message}", alert_type="info"
        )

    def _dynamic_load(self):
        for obj in self.results.values():
            if hasattr(obj, "_dynamic_load"):
                obj._dynamic_load()

    def __panel__(self):
        """Return panel."""
        if self.analysis is None:
            return self.empty_view()

        head = pn.Row(self.period_view)
        return pn.Column(head, self.view)

    def period_view(self):
        raise NotImplementedError("abstract method called: period_view")

    def results_view(self):
        raise NotImplementedError("abstract method called: results_view")

    def results_update(self, obj, kls):
        if obj is not None:
            update = False
            for uuid, ind in obj.items():
                if uuid not in self.results:
                    self.results[uuid] = kls(ind=ind, analysis=self.analysis)
                    if self.this_tab == self.tab:
                        self.results[uuid]._dynamic_load()
                    update = True
            # check for any deleted indices.
            for uuid in self.results:
                if obj.get(uuid, None) is None:
                    self.results[uuid] = None
                    update = True
            if update:
                if self.analysis.indicators is not None:
                    selected = self.analysis.indicators.selected
                else:
                    selected = obj.keys()
                ordered_results = [self.results.get(uuid, None) for uuid in selected]
                self.view.object = pn.Column(*ordered_results)


class ObsAnalysisViewer(AbstractAnalysisViewer):
    this_tab = "ref_period_select"

    # nb: causes infinite loop.
    @pn.depends('analysis.ref_period')
    def period_view(self):
        """Reference period slider."""
        # logger.info(f'Updating period_view: {self.analysis.ref_period}')
        return pn.Param(
            self.analysis.param.ref_period,
            width=600,
            widgets={"period": {"throttled": True}},
        )

    @param.depends("analysis.obs", watch=True)
    def results_view(self):
        logger.debug(f"Observations updated {self.name}")
        if self.analysis is not None:
            self.results_update(self.analysis.obs, IndicatorDAViewer)


class FutAnalysisViewer(AbstractAnalysisViewer):
    """View to select reference period and statistical distributions for indicators."""

    this_tab = "sim_period_select"

    config = param.Dict({})

    @pn.depends('analysis.fut_period')
    def period_view(self):
        """Future period slider."""
        return pn.Param(
            self.analysis.param.fut_period,
            width=600,
            widgets={"period": {"throttled": True}},
        )

    @param.depends("analysis.fut", watch=True)
    def results_view(self):
        logger.debug(f"Futures updated {self.name}")
        if self.analysis is not None:
            self.results_update(self.analysis.fut, IndicatorSimDAViewer)


class IndicatorDAViewer(Viewer):
    """Viewer for the time series and histogram of an indicator."""

    ind = param.ClassSelector(class_=p.IndicatorDA)
    analysis = param.ClassSelector(class_=p.Analysis)

    lw = 600
    rw = 400
    h = 300
    dist_disabled = False

    def __init__(self, *args, **kwargs):
        self.hist_view_caption = pn.pane.Markdown(width=self.rw)

        super().__init__(*args, **kwargs)
        self.plot = pn.pane.Placeholder(
            pn.Row(
                pn.HSpacer(),
                pn.indicators.LoadingSpinner(value=True),
                pn.HSpacer(),
                width=self.lw + self.rw,
                height=self.h,
            ),
            width=self.lw + self.rw,
            height=self.h,
        )

    def period_span(self):
        """Return a VSpan for the period."""
        start = dt.datetime(self.analysis.ref_period[0], 1, 1)
        end = dt.datetime(self.analysis.ref_period[1], 12, 31)
        return hv.VSpan(start, end).opts(color="grey", alpha=0.2)

    @param.depends("analysis.ref_period", "analysis.locale", "ind.dist", watch=True, on_init=True)
    def update_hist_view_caption(self):
        self.hist_view_caption.object =  self.ind.hist_caption

    @property
    def ylabel(self):
        return "\n".join(textwrap.wrap(self.ind.long_name, width=50))

    def ts_view(self):
        """Time series."""
        vspan = self.period_span()
        return (
            self.ind.ts.hvplot(width=self.lw, height=self.h).opts(
                active_tools=["pan"], ylabel=self.ylabel, xlabel=""
            )
            * vspan
        )

    def hist_view(self):
        """Histogram."""
        return self.ind.sample.hvplot.hist(invert=True, normed=True, alpha=0.6).opts(
            width=self.rw, active_tools=["pan"], height=self.h, ylabel="", xlabel=""
        )

    def pdf_view(self):
        """PDF"""
        x = np.linspace(self.ind.sample.min().item(), self.ind.sample.max().item(), 100)
        x = xr.DataArray(
            data=x, dims=(self.ind.data.long_name,), coords={self.ind.data.long_name: x}
        )
        return (
            self.ind.pdf(x)
            .hvplot(invert=True, width=self.rw, height=self.h)
            .opts(active_tools=["pan"], xlabel="", ylabel="")
        )

    def ts_view_caption(self):
        """Caption for the time series."""
        return pn.pane.Markdown(self.ind.ts_caption, width=self.lw)


    def score_view(self):
        items = (
            pn.pane.Markdown(
                "BIC", align=("start", "end"), width_policy="min", sizing_mode="fixed"
            ),
            pn.widgets.TooltipIcon(
                value="Bayesian Information Criterion",
                align=("start", "center"),
                width_policy="min",
                sizing_mode="fixed",
            ),
            pn.pane.Markdown(
                f"{self.ind.bic:.2f}",
                align=("start", "end"),
                width_policy="min",
                sizing_mode="fixed",
            ),
        )
        return pn.Row(*items)

    def create_plots(self):
        return (
            self.ts_view()
            + (self.hist_view() * self.pdf_view()).opts(show_legend=False)
        ).opts(toolbar="right")

    # not getting called on obs data creation
    @param.depends(
        "ind.dparams", "ind.locale", watch=True
    )  # und.period always changes ind.dparams due to param.trigger()
    def _update_plots(self):
        logger.debug(f"Updating plots: {self.name}, {self.ind.description}")

        self.plot.object = pn.pane.HoloViews(self.create_plots(), linked_axes=False)

    def _dynamic_load(self):
        if isinstance(self.plot.object, pn.Row):
            self._update_plots()

    # No watch as we ask panel to watch
    @param.depends("ind.locale")
    def draw(self):
        """Return panel"""
        head = pn.Row(
            pn.pane.Markdown(f"## {self.ind.description}", width=self.lw),
            pn.Param(
                self.ind.param.dist,
                width=self.rw,
                widgets={"dist": {"disabled": self.dist_disabled}},
            ),
        )

        return pn.Column(
            head,
            self.plot,
            pn.Row(self.ts_view_caption(), self.hist_view_caption),
        )

    def __panel__(self):
        return pn.Column(self.draw)


class IndicatorSimDAViewer(IndicatorDAViewer):
    ind = param.ClassSelector(class_=p.IndicatorDA)
    dist_disabled = True

    def period_span(self):
        """Return a VSpan for the period."""
        start = dt.datetime(self.analysis.fut_period[0], 1, 1)
        end = dt.datetime(self.analysis.fut_period[1], 12, 31)
        return hv.VSpan(start, end).opts(color="grey", alpha=0.2)

    @param.depends("analysis.fut_period", "analysis.locale", watch=True, on_init=True)
    def update_hist_view_caption(self):
        self.hist_view_caption.object =  self.ind.hist_caption

    def ts_view(self):
        """Time series of enveloppe percentiles."""
        vspan = self.period_span()
        # For some reason, area doesn't work with xr.Dataset
        per = self.ind.experiment_percentiles(per=[10, 50, 90])
        color = [colors[key] for key in per.experiment_id.values]

        graph = (
            per.to_dataframe().hvplot.area(
                x="time",
                y=f"{self.ind.data.name}_p10",
                y2=f"{self.ind.data.name}_p90",
                stacked=False,
                alpha=0.2,
                by="experiment_id",
                color=color,
                muted_alpha=0,
            )
            * per.hvplot.line(
                x="time",
                y=f"{self.ind.data.name}_p50",
                by="experiment_id",
                color=color,
                muted_alpha=0,
            )
            * self.ind.obs.ts.hvplot(
                width=self.lw, color="k", label="Observations", muted_alpha=0
            )
        ).redim.label(**{f"{self.ind.data.name}_p10": f"{self.ind.data.long_name}"})

        return (graph * vspan).opts(
            active_tools=["pan"],
            legend_position="top_left",
            width=self.lw,
            height=self.h,
            ylabel=self.ylabel,
        )

    def create_plots(self):
        return (self.ts_view() + self.pdf_view()).opts(toolbar="right")


class HazardMatrixViewer(Viewer):
    hm = param.ClassSelector(class_=p.HazardMatrix)

    table = param.Parameter()

    # Notification
    _oob_not = param.Parameter()

    # No watch as panel will watch it himself
    @param.depends("hm.locale")
    def draw_table(self):
        NF = NumberFormatter
        fmt = {
            "obs_sf": NF(format="0.%"),
            "ref_sf": NF(format="0.%"),
            "fut_sf": NF(format="0.%"),
            "ratio": NF(format="0.%"),
            "value": NF(format="0.00"),
            "obs_t": NF(format="0.00"),
        }

        # Tooltips - only work when table is "served" by the server, not when it's a standalone widget in a nb
        tips = self.hm.docs

        # Column widths
        iw = 50
        ow = 120
        tw = {
            "xid": 250,
            "descr": 350,
            "value": iw,
            "obs_t": iw,
            "obs_sf": ow,
            "ref_sf": ow,
            "fut_sf": ow,
            "ratio": ow,
            "add": 20,
            "remove": 20,
        }

        # So I wasted a lot of time trying to make a multi-column dataframe, but a lot of things were broken or
        # shaky, such as the titles and the callbacks. So the solution for now is to two have two separate flat tables,
        # one serving as a header for the categories, and one for the actual data.
        header_df = pd.DataFrame(columns=[v for v in self.hm._label.keys()])
        # Header width
        hw = {cat: sum([tw[k] for k in keys]) for cat, keys in self.hm._levels.items()}

        stylesheets = [":host .tabulator {font-size: 12px;}"]
        header = pn.widgets.Tabulator(
            header_df,
            show_index=False,
            theme="bootstrap5",
            widths=hw,
            titles=self.hm.titles,
            stylesheets=stylesheets,
            sortable=False,
            header_tooltips=tips,
        )

        # Buttons to add and remove rows
        buttons = {
            "add": '<i class="fa fa-plus"></i>',
            "remove": '<i class="fa fa-minus"></i>',
        }

        # Styling to highlight the editable columns on hover
        editable = "{outline-color: #3246a8; outline-width: 2px; outline-style: solid; outline-offset: -2px;}"
        stylesheets = stylesheets + [
            f':host div.tabulator-row:hover [tabulator-field="{key}"] {editable}'
            for key in ["value", "obs_t", "descr"]
        ]

        self.table = pn.widgets.Tabulator(
            self.hm.df,
            editors={
                "value": {"type": "number"},
                "obs_t": {"type": "number", "step": 1},
            },
            selectable=True,
            show_index=False,
            formatters=fmt,
            theme="bootstrap5",
            titles=self.hm.titles,
            widths=tw,
            stylesheets=stylesheets,
            buttons=buttons,
            sortable={
                "descr": False,
                "long_name": False,
                "obs_sf": True,
                "ref_sf": False,
                "fut_sf": False,
                "ratio": True,
                "value": False,
                "obs_t": False,
                "add": False,
            },
            header_tooltips=tips,
        )
        self.table.on_edit(self.on_edit)
        self.table.on_click(self.on_click)

        # In theory, requires panel 1.4.3 to work correctly. In practice, this version breaks the GUI.
        # table.style.background_gradient(
        #     cmap="coolwarm", subset="ratio", vmin=0, vmax=2
        # )

        return pn.Column(header, self.table)

    @param.depends("hm.matrix", watch=True)
    def update_table(self, event=None):
        if hasattr(self, "table"):
            self.table.value = self.hm.df

    def on_edit(self, event):
        ht = self.hm.get_ht(event.row)
        ht.on_edit(event)
        self._notify_out_of_bounds(ht)
        # self.table.value = self.hm.df
        self.update_table()
        # patch = {}
        # for key, val in ht.values.to_dict().items():
        #     if key[0] != "id":
        #         patch[key] = [(i, val)]
        # # Bokeh fails
        # table.patch(patch, as_index=False)

    def on_click(self, event):
        self.hm.on_click(event)
        # self.update_table()
        # self.table.value = self.hm.df

    def _notify_out_of_bounds(self, ht):
        if ht.obs_t is not None and ht.input == "T":
            vmin, vmax = ht.param.obs_t.softbounds
            if ht.obs_t < vmin or ht.obs_t > vmax:
                if self._oob_not is None or self._oob_not._destroyed:
                    self._oob_not = pn.state.notifications.info(
                        "Le temps de retour devrait être une valeur entre 2 et 100 ans.",
                        duration=4000,
                    )
        elif ht.value is not None and ht.input == "X":
            vmin, vmax = ht.param.value.softbounds
            if ht.value < vmin or ht.value > vmax:
                if self._oob_not is None or self._oob_not._destroyed:
                    self._oob_not = pn.state.notifications.info(
                        f"La valeur seuil devrait être entre {vmin:.2f} et {vmax:.2f}, pour correspondre à un temps de retour d'entre 2 et 100 ans.",
                        duration=4000,
                    )

    def __panel__(self):
        return pn.Column(self.draw_table)


class Application(pn.viewable.Viewer):
    global_ = param.ClassSelector(class_=p.Global)

    # Parameters
    station = param.ClassSelector(class_=p.Station)
    indicators = param.ClassSelector(class_=p.IndicatorList)
    analysis = param.ClassSelector(default=None, class_=p.Analysis)
    hazmat = param.ClassSelector(class_=p.HazardMatrix)
    site = param.ClassSelector(class_=p.Site)
    map_param = param.ClassSelector(class_=p.Map)

    layout = param.Parameter()
    menu = param.Parameter()
    text = param.ClassSelector(class_=Text)

    config = param.Dict()
    views = param.Dict({})

    view_state = param.Dict({})

    def __init__(self, **params):
        super().__init__(**params)
        if self.analysis is None:
            with param.parameterized.discard_events(self):
                self.analysis = p.Analysis(
                    indicators=self.indicators, locale=self.global_.param.locale
                )
                self.hazmat = p.HazardMatrix(
                    analysis=self.analysis, locale=self.global_.param.locale
                )
        self.download_grp = pn.pane.Placeholder()
        self.text = Text(locale=self.global_.param.locale)

        self.layout = pn.Tabs(
            *self.dash_init(),
            sizing_mode="stretch_width",
            active=self.global_.tab_index,
            dynamic=False,
        )

    def make_menu(self):
        steps = []
        for key, conf in self.config["steps"].items():
            steps.append(
                (
                    conf["name"][self.global_.locale],
                    f'*{conf["description"][self.global_.locale]}*'
                    + "\n\n"
                    + conf["help"][self.global_.locale],
                )
            )
        return pn.Accordion(*steps, active=self.global_.sidebar_tab)

    @param.depends("menu.active", watch=True)
    def change_sidebar_tab(self):
        self.global_.sidebar_tab = self.menu.active

    @param.depends("global_.tab", watch=True)
    def change_tab(self):
        tab_index = self.global_.tab_index

        if tab_index != self.layout.active and tab_index < len(self.layout):
            if hasattr(self.layout[self.layout.active][1], "_dynamic_unload"):
                self.layout[self.layout.active][1]._dynamic_unload()
            if hasattr(self.layout[tab_index][1], "_dynamic_load"):
                self.layout[tab_index]._dynamic_load()
            self.layout.active = tab_index
        elif tab_index >= len(self.layout):
            self.global_.tab = self.global_.ind_to_tab(self.layout.active)

    @pn.depends("layout.active", watch=True)
    def tab_changed(self):
        logger.debug(f"Tab changed: {self.layout.active}")
        tab_index = self.layout.active
        tab_name = self.global_.ind_to_tab(tab_index)

        if self.global_.tab != tab_name and tab_name in self.global_.tabs:
            old_tab_name = self.global_.tab

            if hasattr(self.views[old_tab_name], "_dynamic_unload"):
                self.views[old_tab_name]._dynamic_unload()

            if hasattr(self.views[tab_name], "_dynamic_load"):
                self.views[tab_name]._dynamic_load()

            self.global_.tab = tab_name

    @param.depends("global_.locale")
    def sidebar(self):
        """Left collapsible sidebar showing the different steps."""
        now = dt.datetime.now()
        st = self.config["sidebar_title"][self.global_.locale]
        sidebar_title = pn.pane.Markdown(f"# {st}")
        self.menu = self.make_menu()

        download_results = LocalizedFileDownload(
            callback=self.export_results,
            filename=f"OuranosAppIng_{now:%Y%m%dT%H%M}.zip",
            button_type="primary",
            label="Exporter les résultats",
            locale=self.global_.locale,
            _value={
                "label": {"fr": "Exporter les résultats", "en": "Export the results"}
            },
            icon="download",
        )
        download_request = LocalizedFileDownload(
            callback=self.export_request,
            filename=f"OuranosAppIng_req_{now:%Y%m%dT%H%M}.json",
            button_type="primary",
            label="Exporter la requête",
            locale=self.global_.locale,
            _value={"label": {"fr": "Exporter la requête", "en": "Export the request"}},
            icon="download",
        )
        self.download_grp.object = pn.Column(
            pn.pane.Markdown(self.text.download),
            pn.Row(
                download_results,
                pn.widgets.TooltipIcon(
                    value="Un fichier JSON compressé contenant tous les résultats et calculs intermédiaires.",
                    max_width=25,
                ),
            ),
            pn.Row(
                download_request,
                pn.widgets.TooltipIcon(
                    value="Un fichier JSON contenant le minimum pour refaire les calculs, ici ou par un appel API.",
                    max_width=25,
                ),
            ),
        )
        import_request = pn.widgets.FileInput(accept="*.json")
        import_request.param.watch(self.import_request, ["value"])
        import_grp = pn.Column(
            pn.pane.Markdown(self.text.load_json),
            pn.Row(
                import_request,
                pn.widgets.TooltipIcon(
                    value="Charger une requête JSON telle que sauvée par le bouton plus haut pour reprendre ou revoir les calculs d'une session précédente.",
                    max_width=25,
                ),
            ),
        )

        # self.layout.link(self.menu, callbacks={"active": self.callback})
        return pn.Column(
            sidebar_title,
            self.menu,
            pn.layout.Divider(),
            self.download_grp,
            pn.layout.Divider(),
            import_grp,
        )

    def make_help(self, step):
        conf = self.config["steps"][step]
        # tt = pn.widgets.TooltipIcon(value=conf["help"][self.locale], align=("start", "start"), width=50)
        html = (
            """<details open><summary>Instructions</summary>"""
            + conf["help"][self.global_.locale]
            + "</details>"
        )
        return pn.Row(pn.pane.Markdown(html), width_policy="max")

    def make_dash_item(self, step, kls, **kwds):
        """Return the name of the dashboard item and its content."""
        try:
            conf = self.config["steps"][step]
            if "view_config" in conf:
                with open(CONFIG_DIR / conf["view_config"]) as fh:
                    config = yaml.safe_load(fh)
                    kwds["config"] = config

            header = conf["header"][self.global_.locale]
            view = kls(**kwds)
            self.views[step] = view

            if self.global_.tab == step and hasattr(view, "_dynamic_load"):
                pn.state.onload(view._dynamic_load)
            return header, pn.Column(view)

        except Exception as err:
            logger.error(f"make_dash_item failed for step {step} with {err}")
            print(traceback.format_exc())
            raise

    def dash_init(self):
        """Main dashboard."""
        # main_title = pn.pane.Markdown(f"# {self.dash_title}")

        return (
            self.make_dash_item(
                "station_select",
                StationViewer,
                station=self.param.station,
                site=self.param.site,
                map_param=self.param.map_param,
            ),
            self.make_dash_item(
                "indicator_select", IndicatorListViewer, inds=self.indicators
            ),
        )

    @param.depends("analysis.obs", watch=True)
    def _add_obs_tab(self):
        """Load the computed indicators and create analysis object."""
        if "ref_period_select" not in self.views:
            self.layout.append(
                self.make_dash_item(
                    "ref_period_select",
                    ObsAnalysisViewer,
                    analysis=self.analysis,
                    tab=self.global_.param.tab,
                )
            )

    @param.depends("analysis.fut", watch=True)
    def _add_sim_tab(self):
        """Load the computed indicators and create analysis object."""
        if "fut_period_select" not in self.views:
            self.layout.append(
                self.make_dash_item(
                    "fut_period_select",
                    FutAnalysisViewer,
                    analysis=self.analysis,
                    tab=self.global_.param.tab,
                )
            )

    @param.depends("_add_sim_tab", watch=True)
    def _add_hazmat_tab(self):
        if "hazard_threshold_select" not in self.views:
            self.layout.append(
                self.make_dash_item(
                    "hazard_threshold_select", HazardMatrixViewer, hm=self.hazmat
                )
            )
            # self.matrix_updated = True

    def lang_switcher(self):
        return pn.Param(
            self.global_.param.locale,
            width=150,
            widgets={"locale": {"widget_type": pn.widgets.Select}},
        )

    def docs(self):
        styles = {"text-decoration": "none"}
        stylesheet = """
        <style>
        a {
            text-decoration: none;
            color: white;
        }
        </style>
        """
        en = pn.pane.HTML(f"""{stylesheet}<h1><a href="docs/en/index.html">Documentation (en)</a></h1>""")
        fr = pn.pane.HTML(f"""{stylesheet}<h1><a href="docs/fr/index.html">Documentation (fr)</a></h1>""")
        return en, fr

    @staticmethod
    def callback(target, event):
        target.active = [event.new]

    def to_dict(self, mode="results"):
        import portail_ing

        if mode == "results":
            meta = {
                "date": dt.datetime.now().isoformat(),
                "locale": self.global_.locale,
                "version": portail_ing.__version__,
                "contact": "Sarah-Claude Bourdeau-Goulet <bourdeau-goulet.sarah-claude@ouranos.ca>",
                "reference": "TODO",
                "license": "CC-BY-4.0",
            }

            out = {
                **meta,
                "stations": self.station.to_dict(),
                "indicators": self.indicators.to_dict(),
                "analysis": self.analysis.to_dict(),
                "matrix": self.hazmat.to_dict(),
            }
        elif mode == "request":
            out = {
                "indicators": self.indicators.to_dict(mode="request"),
                "stations": {
                    vv: sid
                    for vv, sid in self.station.station_id.items()
                    if sid is not None
                },
                "analysis": self.analysis.to_dict(mode="request"),
                "hazards": self.hazmat.to_dict(mode="request"),
            }
        return out

    def export_results(self):
        """Export the configuration as a json file to a zip archive."""
        self.download_grp.loading = True
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr(
                "config.json",
                json.dumps(
                    self.to_dict(mode="results"),
                    indent=4,
                    default=str,
                    ensure_ascii=False,
                ),
            )
        data.seek(0)
        self.download_grp.loading = False
        return data

    def export_request(self):
        """Export the minimal configuration needed to reproduce the results, as a json file."""
        self.download_grp.loading = True
        data = io.StringIO()
        data.write(
            json.dumps(
                self.to_dict(mode="request"), indent=4, default=str, ensure_ascii=False
            )
        )
        data.seek(0)
        self.download_grp.loading = False
        return data

    def import_request(self, event):
        """Import a request and load the application."""
        if event.new is None:
            return

        self.layout.loading = True
        try:
            data = json.loads(event.new.decode())
        except Exception:
            pn.state.notifications.error("Le fichier de requête semble invalide.")
            print(traceback.format_exc())
            self.layout.loading = False
            event.obj.clear()
            return

        if (
            len(data["indicators"]) + len(self.indicators.selected)
        ) > global_config.MAX_INDICATORS:
            pn.state.notifications.error(
                f"Le nombre d'indicateurs est limité à {global_config.MAX_INDICATORS}, "
                f"mais la requête en contient {len(data['indicators'])} "
                f"et il y en a déjà {len(self.indicators.selected)} dans la session."
            )
            self.layout.loading = False
            event.obj.clear()
            return

        try:
            # Stations, done this way for 2 reasons:
            #  1. Keep the stations for non-updated variables that might correspond to already computed indicators
            #  2. Trigger station_id
            station_id = self.station.station_id | data["stations"]
            self.station.station_id = station_id

            # Indicators
            uuids = self.indicators.select_from_list(data["indicators"])

            # Compute
            self.indicators.post_all_requests()
            while self.indicators.monitor_jobs() is False:
                # TODO Timeout ?
                time.sleep(2)

            # Analysis
            params = {
                k: tuple(data["analysis"][k])
                for k in ["ref_period", "fut_period"]
                if k in data["analysis"]
            } | {
                k: dict(zip(uuids, data["analysis"][k]))
                for k in ["metric", "dist"]
                if k in data["analysis"]
            }
            self.analysis.update_params(**params)

            # Hazards
            hazarddict = dict(zip(uuids, data["hazards"]))
            self.hazmat.from_dict(hazarddict)
        except Exception:
            pn.state.notifications.error(
                "L'importation de la requête a échoué. Désolé."
            )
            print(traceback.format_exc())
        finally:
            self.layout.loading = False
            event.obj.clear()

    def __panel__(self) -> Viewable:
        logger.info("Drawing template")
        header = pn.Row(pn.HSpacer(), *self.docs(), self.lang_switcher())
        title = self.config["title"][self.global_.locale]
        out = pn.template.VanillaTemplate(
            title=title,
            header=header,
            sidebar=pn.Column(self.sidebar),
            sidebar_width=400,
            main=pn.Column(self.layout),
        )
        # debug = pn.widgets.Debugger(
        #     logger_names=["panel.myapp"], level=logging.INFO, sizing_mode="stretch_both"
        # )
        return out


# Custom styling

checkmark = """
:host(.solid) .bk-btn-group button.bk-active.bk-btn[disabled] {
  opacity: 1;
}
:host(.solid) .bk-btn-group button.bk-btn.bk-btn-default.bk-btn-default {
  background-color: #d2322d;
  color: #fff;
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
:host(.solid) .bk-btn-group button.bk-btn.bk-btn-default.bk-btn-default.bk-active {
  background-color: #47a447;
  color: #fff;
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
:host(.solid) .bk-btn-group .bk-btn {
  max-width: 60px;
}
"""
