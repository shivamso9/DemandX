"""
Plugin : DP210CalculateBacktestStatLevelAccuracy
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    StatGrains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    BackTestCyclePeriod - 2,4,6

Input Queries:
    StatActual : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row, ({Measure.[Stat Actual]}) on column;

    StatFcstL1Lag1Backtest : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row, ({Measure.[Stat Fcst L1 Lag1 Backtest]}) on column;

    StatFcstL1LagNBacktest : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Location].[Stat Location] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Item].[Stat Item] * [Demand Domain].[Stat Demand Domain] * [Time].[Partial Week] * [Account].[Stat Account] ) on row,
({Measure.[Stat Fcst L1 LagN Backtest]}) on column;

    StatFcstL1WLagBacktest : Select ([Version].[Version Name] * [Time].[Week] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row, ({Measure.[Stat Fcst L1 W Lag Backtest]}) on column;

    StatFcstL1MLagBacktest : Select ([Version].[Version Name] * [Time].[Month] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row, ({Measure.[Stat Fcst L1 M Lag Backtest]}) on column;

    StatFcstL1PMLagBacktest : Select ([Version].[Version Name] * [Time].[Planning Month] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row, ({Measure.[Stat Fcst L1 PM Lag Backtest]}) on column;

    StatFcstL1QLagBacktest : Select ([Version].[Version Name] * [Time].[Quarter] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row, ({Measure.[Stat Fcst L1 Q Lag Backtest]}) on column;

    StatFcstL1PQLagBacktest : Select ([Version].[Version Name] * [Time].[Planning Quarter] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row, ({Measure.[Stat Fcst L1 PQ Lag Backtest]}) on column;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

Output Variables:
    StatFcstL1AbsErrorBacktest
    StatFcstL1LagAbsErrorBacktest
    StatFcstL1LagNAbsErrorBacktest
    StatFcstL1WLagAbsErrorBacktest
    StatFcstL1MLagAbsErrorBacktest
    StatFcstL1PMLagAbsErrorBacktest
    StatFcstL1QLagAbsErrorBacktest
    StatFcstL1PQLagAbsErrorBacktest

Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP210CalculateBacktestStatLevelAccuracy import main

logger = logging.getLogger("o9_logger")


# Function Calls
StatActual = O9DataLake.get("StatActual")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatFcstL1Lag1Backtest = O9DataLake.get("StatFcstL1Lag1Backtest")
StatFcstL1LagNBacktest = O9DataLake.get("StatFcstL1LagNBacktest")
StatFcstL1WLagBacktest = O9DataLake.get("StatFcstL1WLagBacktest")
StatFcstL1MLagBacktest = O9DataLake.get("StatFcstL1MLagBacktest")
StatFcstL1PMLagBacktest = O9DataLake.get("StatFcstL1PMLagBacktest")
StatFcstL1QLagBacktest = O9DataLake.get("StatFcstL1QLagBacktest")
StatFcstL1PQLagBacktest = O9DataLake.get("StatFcstL1PQLagBacktest")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
TimeDimension = O9DataLake.get("TimeDimension")

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
    StatFcstL1AbsErrorBacktest,
    StatFcstL1LagAbsErrorBacktest,
    StatFcstL1LagNAbsErrorBacktest,
    StatFcstL1WLagAbsErrorBacktest,
    StatFcstL1MLagAbsErrorBacktest,
    StatFcstL1PMLagAbsErrorBacktest,
    StatFcstL1QLagAbsErrorBacktest,
    StatFcstL1PQLagAbsErrorBacktest,
) = main(
    StatActual=StatActual,
    StatFcstL1Lag1Backtest=StatFcstL1Lag1Backtest,
    StatFcstL1LagNBacktest=StatFcstL1LagNBacktest,
    StatFcstL1WLagBacktest=StatFcstL1WLagBacktest,
    StatFcstL1MLagBacktest=StatFcstL1MLagBacktest,
    StatFcstL1PMLagBacktest=StatFcstL1PMLagBacktest,
    StatFcstL1QLagBacktest=StatFcstL1QLagBacktest,
    StatFcstL1PQLagBacktest=StatFcstL1PQLagBacktest,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    StatGrains=StatGrains,
    StatBucketWeight=StatBucketWeight,
    BackTestCyclePeriod=BackTestCyclePeriod,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    df_keys={},
)

O9DataLake.put("StatFcstL1AbsErrorBacktest", StatFcstL1AbsErrorBacktest)

O9DataLake.put("StatFcstL1LagAbsErrorBacktest", StatFcstL1LagAbsErrorBacktest)

O9DataLake.put("StatFcstL1LagNAbsErrorBacktest", StatFcstL1LagNAbsErrorBacktest)

O9DataLake.put("StatFcstL1WLagAbsErrorBacktest", StatFcstL1WLagAbsErrorBacktest)
O9DataLake.put("StatFcstL1MLagAbsErrorBacktest", StatFcstL1MLagAbsErrorBacktest)
O9DataLake.put("StatFcstL1PMLagAbsErrorBacktest", StatFcstL1PMLagAbsErrorBacktest)
O9DataLake.put("StatFcstL1QLagAbsErrorBacktest", StatFcstL1QLagAbsErrorBacktest)
O9DataLake.put("StatFcstL1PQLagAbsErrorBacktest", StatFcstL1PQLagAbsErrorBacktest)
