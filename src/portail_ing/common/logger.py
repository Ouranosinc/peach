import logging
import os
import sys

FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
default_level = os.getenv("PANEL_ADMIN_LOG_LEVEL", "INFO")


def get_logger(name, format_=FORMAT, level=default_level):
    logger = logging.getLogger(name)

    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setStream(sys.stdout)
    formatter = logging.Formatter(format_)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    logger.setLevel(level)
    logger.info("Logger successfully configured")
    return logger
