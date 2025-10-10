"""
# TODO : Change plugin name here
Plugin : DP301NPIDisaggRegionToCountry
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Input Queries:
//LikeItem
Select ([Version].[Version Name].[CurrentWorkingView] * [Item].[Stat Item] * [Channel].[Stat Channel] * [Account].[Stat Account] * [Location].[Stat Location] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] ) on row,
({Measure.[Like Item Selected]}) on column;

//ActualLikeItem
Select ( [Version].[Version Name].[CurrentWorkingView] * [Item].[Stat Item] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Location Country] * [Time].[Partial Week] ) on row,({Measure.[Actual]}) on column.filter(Measure.[Actual]>0);

//Active
Select ([Version].[Version Name].[CurrentWorkingView] * [Item].[Stat Item] * [Channel].[Stat Channel] *[Location].[Stat Location] * [Location].[Location Country] ) on row, ({Measure.[Active]}) on column;

//Last13Weeks
select ([Time].[Partial Week].filter(#.Key<&CurrentWeek.element(0).Key && #.Key>=&CurrentWeek.element(0).LeadOffset(-13).Key) );

//NPIForecast
Select ([Version].[Version Name].[CurrentWorkingView] * [Item].[Stat Item] * [Channel].[Stat Channel] * &AllWeeks.Filter(#.Key >= &CurrentWeek.element(0).LeadOffset(0).Key).relatedmembers([Partial Week]) * [Account].[Stat Account] * [Location].[Stat Location] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] ) on row, ({Measure.[NPI Fcst], Measure.[Stat Fcst TLG Internal] }) on column.filter(Measure.[NPI Assortment] == 1);

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
NPIForecast_df = O9DataLake.get("NPIForecast")
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

from helpers.DP301NPIDisaggRegionToCountry import main

DisaggregationOP = main(
    Actual_df,
    LikeItem_df,
    NPIForecast_df,
    Last13Weeks_df,
    Active_df,
    df_keys=df_keys,
)
O9DataLake.put("DisaggregationOP", DisaggregationOP)
