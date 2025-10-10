"""
Plugin : DP079ForecastRaw
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    MeasureNames - Sell Out Stat Fcst New FND BB,Sell Out Stat Fcst KAF BB New FND BB,Sell In FND BB,Cannibalization Impact FND BB,Cannibalization Impact Published FND BB,Cannibalized Stat Fcst L0 FND BB,NPI Fcst FND BB,NPI Fcst Published FND BB,Stat Fcst FND BB,Stat Fcst L0 FND BB

Input Queries:
    SellInForecast - Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Channel].[Planning Channel] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] ) on row,  ({Measure.[Cannibalization Impact FND BB], Measure.[Cannibalization Impact Published FND BB], Measure.[Cannibalized Stat Fcst L0 FND BB], Measure.[NPI Fcst FND BB], Measure.[NPI Fcst Published FND BB], Measure.[Stat Fcst FND BB], Measure.[Stat Fcst L0 FND BB]}) on column;

    KeyFigures - Select ([Version].[Version Name] * [Data Object].[Data Object] ) on row,  ({Measure.[Include in Forecast Realignment], Measure.[One Time Realignment]}) on column;

    SellOutForecast - Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain]) on row,  ({Measure.[Sell Out Stat Fcst New FND BB],Measure.[Sell Out Stat Fcst KAF BB New FND BB],Measure.[Sell In FND BB]}) on column;

Output Variables:
    RawSellInForecast
    RawSellOutForecast

Slice Dimension Attributes:

"""

import logging

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

from o9_common_utils.O9DataLake import O9DataLake

logger = logging.getLogger("o9_logger")

import threading

from o9Reference.common_utils.o9_memory_utils import _get_memory

# Function Calls
SellInForecast = O9DataLake.get("SellInForecast")
KeyFigures = O9DataLake.get("KeyFigures")
SellOutForecast = O9DataLake.get("SellOutForecast")

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

from helpers.DP079ForecastRaw import main

(
    RawSellInForecast,
    RawSellOutForecast,
) = main(
    MeasureNames=MeasureNames,
    SellInForecast=SellInForecast,
    KeyFigures=KeyFigures,
    SellOutForecast=SellOutForecast,
    df_keys=df_keys,
)
O9DataLake.put("RawSellInForecast", RawSellInForecast)
O9DataLake.put("RawSellOutForecast", RawSellOutForecast)
