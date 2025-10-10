"""
Plugin : DP245APHandshakeRateofSale_Pyspark.

Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Input Queries:
    HMLInput : 514-1 HML Forecast Components
    SellingSeasonInput : 025 Selling Season Week Association
    Ranging : Select ( [Version].[Version Name] * $$Retail_AP_RP_Product_Cluster_NN * $$Retail_AP_FD_Common_NN * { Measure.[Assorted SC L]}) where {Measure.[Assorted SC L]==true};

Output Variables:
    Aggregated Forecast - 010 DP ROS
    ROS - 514 HML Forecast Components
"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.pyspark.DP245APHandshakeRateofSale_Pyspark import main

# Initialize logging
logger = logging.getLogger("o9_logger")

HML = O9DataLake.get("HML")
Selling_Season_week_association = O9DataLake.get("SellingSeason")
Ranging = O9DataLake.get("Ranging")

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.

back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

aggregated_forecast, ros = main(
    HML=HML,
    Selling_Season_week_association=Selling_Season_week_association,
    Ranging=Ranging,
    spark=spark,
)

O9DataLake.put("AggregatedForecast", aggregated_forecast)
O9DataLake.put("ROS", ros)
