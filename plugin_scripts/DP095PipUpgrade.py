"""
    Plugin : DP095PipUpgrade
    Version : 2025.08.00
    Maintained by : dpref@o9solutions.com

"""

import logging

logger = logging.getLogger("o9_logger")


from helpers.utils_pip_upgradation import upgrade_package

upgrade_package("statsforecast", "1.7.8")
