# Script to run before running gui_serve.py, with panel serve --warm warm.py

import panel as pn

from portail_ing.common.logger import get_logger

logger = get_logger("app")
logger.info("Loading warm.py")

import xclim  # noqa: F401, E402
import xscen  # noqa: F401, E402

from portail_ing.common import config as conf  # noqa: E402

pn.extension(
    "terminal",
    "tabulator",
    "ipywidgets",
    notifications=True,
    sizing_mode="stretch_width",
    throttled=True,
)

# console_output='disable')
pn.state.cache["ic"] = conf.read_indicator_config()
pn.state.cache["ac"] = conf.read_application_config()
pn.state.cache["df_sta"] = conf.read_stations()

logger.info("warm.py loaded!")
