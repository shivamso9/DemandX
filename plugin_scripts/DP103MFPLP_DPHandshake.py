"""
Plugin : DP103MFPLP_DPHandshake

Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Input Queries:

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter] *  Time.[Year]*  Time.[Week Name]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key} {Time.[Year], Key} {Time.[Week Name], Key};

    LPTyasp = Select ([Version].[Version Name] * $$Retail_MFP_TY_LP_Time_NN * $$Retail_MFP_Common_NN * $$Retail_MFP_TY_LP_Product_Location_NN ) on row,
    ({Measure.[LP TY ASP], Measure.[LP ASP]}) on column;

    RegionMaster : Select ([Region].[Planning Region]) on row, () on column;

    PnLMaster : Select ([PnL].[Planning PnL]) on row, () on column;

    AccountMaster : Select ([Account].[Planning Account])  on row, () on column;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

Output Variables:
    TYASP

Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP103MFPLP_DPHandshake import main

logger = logging.getLogger("o9_logger")

# Function Calls
TimeDimension = O9DataLake.get("TimeDimension")
LPTyasp = O9DataLake.get("LPTyasp")
RegionMasterData = O9DataLake.get("RegionMaster")
PnlMasterData = O9DataLake.get("PnLMaster")
AccountMasterData = O9DataLake.get("AccountMaster")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")


# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.
back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

TYASP = main(
    time_dimension=TimeDimension,
    lptyasp_input=LPTyasp,
    region_master=RegionMasterData,
    account_master=AccountMasterData,
    pnl_master=PnlMasterData,
    current_time_period=CurrentTimePeriod,
    df_keys=df_keys,
    logger=logger,
)

O9DataLake.put("TYASP", TYASP)
