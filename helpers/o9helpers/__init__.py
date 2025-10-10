"""
Module level info.

Prints package description when helpers is imported for the first time.
"""

__version__ = "2025.04.00"
import logging

import pkg_resources

try:
    __package_description__ = pkg_resources.get_distribution("helpers").get_metadata("METADATA")
except Exception:
    __package_description__ = ""

logger = logging.getLogger("o9_logger")

logger.info("Importing module : {}, version : {}".format(__name__, __version__))
logger.info(f"DemandPlanningPlugins (helpers) pkg desc: {__package_description__}")
