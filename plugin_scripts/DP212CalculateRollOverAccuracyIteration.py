"""
Plugin : DP212CalculateRollOverAccuracyIteration
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    StatGrains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    AccuracyWindow - 6

Input Queries:
    StatActual : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row, ({Measure.[Stat Actual]}) on column;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    StatFcstL1WLag : Select ([Version].[Version Name] * [Item].[Stat Item] * [Channel].[Stat Channel] * [Time].[Week] * [Location].[Stat Location] * [Account].[Stat Account] * [Demand Domain].[Stat Demand Domain] * [Lag].[Lag] * [Forecast Iteration].[Forecast Iteration] * [PnL].[Stat PnL] * [Region].[Stat Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst L1 W Lag]}) on column;

    StatFcstL1MLag : Select ([Version].[Version Name] * [Item].[Stat Item] * [Channel].[Stat Channel] * [Time].[Month] * [Location].[Stat Location] * [Account].[Stat Account] * [Demand Domain].[Stat Demand Domain] * [Lag].[Lag] * [Forecast Iteration].[Forecast Iteration] * [PnL].[Stat PnL] * [Region].[Stat Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst L1 M Lag]}) on column;

    StatFcstL1PMLag : Select ([Version].[Version Name] * [Item].[Stat Item] * [Channel].[Stat Channel] * [Time].[Planning Month] * [Location].[Stat Location] * [Account].[Stat Account] * [Demand Domain].[Stat Demand Domain] * [Lag].[Lag] * [Forecast Iteration].[Forecast Iteration] * [PnL].[Stat PnL] * [Region].[Stat Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst L1 PM Lag]}) on column;

    StatFcstL1QLag : Select ([Version].[Version Name] * [Item].[Stat Item] * [Channel].[Stat Channel] * [Time].[Quarter] * [Location].[Stat Location] * [Account].[Stat Account] * [Demand Domain].[Stat Demand Domain] * [Lag].[Lag] * [Forecast Iteration].[Forecast Iteration] * [PnL].[Stat PnL] * [Region].[Stat Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst L1 Q Lag]}) on column;

    StatFcstL1PQLag : Select ([Version].[Version Name] * [Item].[Stat Item] * [Channel].[Stat Channel] * [Lifecycle Time].[Planning Quarter] * [Location].[Stat Location] * [Account].[Stat Account] * [Demand Domain].[Stat Demand Domain] * [Lag].[Lag] * [Forecast Iteration].[Forecast Iteration] * [PnL].[Stat PnL] * [Region].[Stat Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst L1 PQ Lag]}) on column;

Output Variables:
    StatFcstL1AbsError
    StatFcstL1LagAbsError
    StatFcstL1WLagAbsError
    StatFcstL1MLagAbsError
    StatFcstL1PMLagAbsError
    StatFcstL1QLagAbsError
    StatFcstL1PQLagAbsError

Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP212CalculateRollOverAccuracyIteration import main

logger = logging.getLogger("o9_logger")


# Function Calls
StatActual = O9DataLake.get("StatActual")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
TimeDimension = O9DataLake.get("TimeDimension")
StatFcstL1WLag = O9DataLake.get("StatFcstL1WLag")
StatFcstL1MLag = O9DataLake.get("StatFcstL1MLag")
StatFcstL1PMLag = O9DataLake.get("StatFcstL1PMLag")
StatFcstL1QLag = O9DataLake.get("StatFcstL1QLag")
StatFcstL1PQLag = O9DataLake.get("StatFcstL1PQLag")

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
    StatFcstL1AbsError,
    StatFcstL1LagAbsError,
    StatFcstL1WLagAbsError,
    StatFcstL1MLagAbsError,
    StatFcstL1PMLagAbsError,
    StatFcstL1QLagAbsError,
    StatFcstL1PQLagAbsError,
) = main(
    StatActual=StatActual,
    StatFcstL1WLag=StatFcstL1WLag,
    StatFcstL1MLag=StatFcstL1MLag,
    StatFcstL1PMLag=StatFcstL1PMLag,
    StatFcstL1QLag=StatFcstL1QLag,
    StatFcstL1PQLag=StatFcstL1PQLag,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    StatGrains=StatGrains,
    StatBucketWeight=StatBucketWeight,
    AccuracyWindow=AccuracyWindow,
    df_keys=df_keys,
)

O9DataLake.put("StatFcstL1AbsError", StatFcstL1AbsError)
O9DataLake.put("StatFcstL1LagAbsError", StatFcstL1LagAbsError)
O9DataLake.put("StatFcstL1WLagAbsError", StatFcstL1WLagAbsError)
O9DataLake.put("StatFcstL1MLagAbsError", StatFcstL1MLagAbsError)
O9DataLake.put("StatFcstL1PMLagAbsError", StatFcstL1PMLagAbsError)
O9DataLake.put("StatFcstL1QLagAbsError", StatFcstL1QLagAbsError)
O9DataLake.put("StatFcstL1PQLagAbsError", StatFcstL1PQLagAbsError)
