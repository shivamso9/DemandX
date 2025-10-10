"""
Plugin : DP006APDPHandhake
Description : This plugin is used to create the Planning Item Customer Group
             from the Ranging, PnL Master, Region Master and Account Master.

Version : 2025.04.00
Maintained by : dpref@o9solutions.com

Input Queries:
    Ranging : Select ([Version].[Version Name] * $$Retail_AP_FD_Common_NN * $$Retail_AP_RP_Product_Cluster_NN ) on row,  ({Measure.[Start Week SC L AP], Measure.[End Week SC L AP]}) on column;
    PnlMaster: select ([PnL].[Planning PnL]);
    RegionMaster: select ([Region].[Planning Region]);
    AccountMaster: select ([Account].[Planning Account]);

Output Variables:
    PlanningItemCustomerGroup
"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.pyspark.DP006DPAPHandshake_pyspark import main

# Initialize logging
logger = logging.getLogger("o9_logger")

Ranging = O9DataLake.get("Ranging")
PnlMaster = O9DataLake.get("PnlMaster")
RegionMaster = O9DataLake.get("RegionMaster")
AccountMaster = O9DataLake.get("AccountMaster")

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}
logger.info("Slice : {}".format(df_keys))

back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

planning_item_cutsomer_group = main(
    Ranging=Ranging,
    PnlMaster=PnlMaster,
    RegionMaster=RegionMaster,
    AccountMaster=AccountMaster,
    spark=spark,
)

O9DataLake.put("PlanningItemCustomerGroup", planning_item_cutsomer_group)
