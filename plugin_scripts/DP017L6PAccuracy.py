"""
Plugin : DP017L6PAccuracy
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params: None

Input Queries:
    AccuracyStat - Select ( [Version].[Version Name] * [Item].[Stat Item] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * &BacktestStatL1HistoryBucket .relatedmembers([Week])    *{Measure.[Stat Fcst Lag1], Measure.[Actual]}) ;
    StatN6PLC - Select ( [Version].[Version Name] * [Item].[Stat Item] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * &AllWeeks.filter(#.Key >= &CurrentWeek.element(0).Key && #.Key <= &CurrentWeek.element(0).LeadOffset(5).Key)   *{Measure.[Stat Fcst LC], Measure.[Stat Fcst L1]}) ;
    AccuracyProduct - Select ( [Version].[Version Name] * [Item].[Planning Item] * &BacktestStatL1HistoryBucket .relatedmembers([Week]) *{Measure.[Stat Fcst Lag1], Measure.[Actual]}) ;
    AccuracyProductCustomer - Select ( [Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * &BacktestStatL1HistoryBucket .relatedmembers([Week])    *{Measure.[Stat Fcst Lag1], Measure.[Actual]}) ;

Output Variables:
    AccuracyOutputStat
    AccuracyOutputProductCustomer
    AccuracyOutputProduct
    N6PFcstLCOutput

Slice Dimension Attributes: None
"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP017L6PAccuracy import main

logger = logging.getLogger("o9_logger")

# Function Calls
AccuracyStat = O9DataLake.get("AccuracyStat")
StatN6PLC = O9DataLake.get("StatN6PLC")
AccuracyProduct = O9DataLake.get("AccuracyProduct")
AccuracyProductCustomer = O9DataLake.get("AccuracyProductCustomer")

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
(
    AccuracyOutputStat,
    AccuracyOutputProduct,
    AccuracyOutputProductCustomer,
) = main(
    AccuracyStat=AccuracyStat,
    StatN6PLC=StatN6PLC,
    AccuracyProduct=AccuracyProduct,
    AccuracyProductCustomer=AccuracyProductCustomer,
    df_keys=df_keys,
)
O9DataLake.put("AccuracyOutputStat", AccuracyOutputStat)
O9DataLake.put("AccuracyOutputProduct", AccuracyOutputProduct)
O9DataLake.put("AccuracyOutputProductCustomer", AccuracyOutputProductCustomer)
