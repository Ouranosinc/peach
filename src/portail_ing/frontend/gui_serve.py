"""Simple GUI to run with panel serve"""

import panel as pn

import portail_ing.frontend.parameters as parameters
import portail_ing.frontend.views as views
from portail_ing.common import config

pn.extension(
    "terminal",
    "tabulator",
    "ipywidgets",
    notifications=True,
    sizing_mode="stretch_width",
    throttled=True,
)

df_sta = pn.state.cache.get("df_sta", None)
if df_sta is None:
    df_sta = config.read_stations()
    pn.state.cache["df_sta"] = df_sta

ic = pn.state.cache.get("ic", None)
if ic is None:
    ic = config.read_indicator_config()
    pn.state.cache["ic"] = ic

ac = pn.state.cache.get("ac", None)
if ac is None:
    ac = config.read_application_config()
    pn.state.cache["ac"] = ac

# Create parameters
gl = parameters.Global(
    sync_url=True,
    locale="fr",
    backend=config.BACKEND_URL,
    tab="station_select",
    tabs=list(ac["steps"].keys()),
    ignored_queries=["backend", "tabs", "sidebar_tab"],
)

site = parameters.Site(
    sync_url=False,
    sync_on_params={gl: {"tab": "station_select"}},
    locale=gl.param.locale,
    ignored_queries=["locale", "enabled", "x", "y"],
)

station = parameters.Station(
    sync_url=False,
    ignored_queries=[
        "df",
        "dependencies",
        "locale",
        "variables",
        "selected_df",
        "station_id",
        "site_df",
        "site",
    ],
    df=df_sta,
    site=site,
    locale=gl.param.locale,
)

map_param = parameters.Map(
    sync_url=False,
    sync_on_params={gl: {"tab": "station_select"}},
    locale=gl.param.locale,
    ignored_queries=["locale", "enabled"],
)

inds = parameters.IndicatorList(
    config=ic,
    locale=gl.param.locale,
    station_id=station.param.station_id,
    backend=gl.param.backend,
)

# To speed up UI testing, set to True
if False:
    # station.pr = "8202250"
    # station.wl = "490"
    station.tas = "7027725"
    # station.idf = "7027725"
    # inds.add("WETDAYS")
    inds.add("TN_MAX")
    inds.add("TN_MIN")
    # inds.add("WL_POT")
    # inds.add("IDF")

app = views.Application(
    global_=gl,
    station=station,
    site=site,
    map_param=map_param,
    indicators=inds,
    config=ac,
)
app.servable()

# if __name__ == "__main__":
# app.servable()

# To develop in a notebook
"""
import os
from importlib import reload
import warnings
os.environ["WORKSPACE"] = "/home/david/src/portail-ing/workspace"
import panel as pn

pn.extension('tabulator', 'terminal', "ipywidgets", sizing_mode="stretch_width", console_output='disable', notifications=True)

from portail_ing.frontend import gui_serve, parameters, views

reload(parameters)
reload(views)
reload(gui_serve)

warnings.filterwarnings('ignore')

gui_serve.app.layout
"""
