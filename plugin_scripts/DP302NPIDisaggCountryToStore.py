"""
# TODO : Change plugin name here
Plugin : DP302NPIDisaggCountryToStore
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Input Queries:
//Output2
select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Partial Week] * [Account].[Planning Account] * [Location].[Location Country] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain]) on row,
({Measure.[TLG Stat Fcst Adj], Measure.[TLG Internal Country Fcst Adj] }) on column;

//LikeItem2
Select ([Version].[Version Name].[CurrentWorkingView] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Account].[Planning Account] * [Location].[Location Country] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] ) on row,
({Measure.[Like Item Selected]}) on column;

//Actual2
Select ( [Version].[Version Name].[CurrentWorkingView] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Time].[Partial Week] ) on row,({Measure.[Actual]}) on column.filter(Measure.[Actual]>0);

//Active
Select ([Version].[Version Name].[CurrentWorkingView] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Location].[Location Country] *[Location].[Planning Location] ) on row, ({Measure.[Active]}) on column;

Output Variables:
DisaggregationOP

"""

import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

# Function Calls
# TODO : Get all input dataframes from o9DataLake
Actual_df = O9DataLake.get("Actual")
LikeItem_df = O9DataLake.get("LikeItem")
NPIForecast_df = O9DataLake.get("Output2")
Last13Weeks_df = O9DataLake.get("Last13Weeks")
Active_df = O9DataLake.get("Active")

# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

# Start a thread to logger.info memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.

back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

from helpers.DP302NPIDisaggCountryToStore import main

DisaggregationOP = main(
    Actual_df,
    LikeItem_df,
    NPIForecast_df,
    Last13Weeks_df,
    Active_df,
    df_keys=df_keys,
)
O9DataLake.put("DisaggregationOP", DisaggregationOP)
