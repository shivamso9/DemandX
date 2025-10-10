"""
Plugin : DP220CalculateLagModelL1
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    StatGrains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    LagWindow - All

Input Queries:
    StatFcstL1 : Select ([Version].[Version Name] * [Time].[Partial Week] * [Account].[Stat Account] * [Item].[Stat Item] * [Location].[Stat Location] * [Region].[Stat Region] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Stat Fcst L1]}) on column;

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] *[Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {[Time].[Quarter], Key} {[Time].[Planning Quarter], Key};

    SelectedPlanningCycle : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    PlanningCycles : select([Planning Cycle].[Planning Cycle Date]) on row, () on column include memberproperties{[Planning Cycle].[Planning Cycle Date],Key};

Output Variables:
    StatFcstL1WLag
    StatFcstL1MLag
    StatFcstL1PMLag
    StatFcstL1QLag
    StatFcstL1PQLag


Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP220CalculateLagModelL1 import main

logger = logging.getLogger("o9_logger")


# Function Calls
StatFcstL1 = O9DataLake.get("StatFcstL1")
SelectedPlanningCycle = O9DataLake.get("SelectedPlanningCycle")
PlanningCycles = O9DataLake.get("PlanningCycles")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
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
    StatFcstL1WLag,
    StatFcstL1MLag,
    StatFcstL1PMLag,
    StatFcstL1QLag,
    StatFcstL1PQLag,
) = main(
    StatFcstL1=StatFcstL1,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    SelectedPlanningCycle=SelectedPlanningCycle,
    PlanningCycles=PlanningCycles,
    TimeDimension=TimeDimension,
    StatGrains=StatGrains,
    LagWindow=LagWindow,
    df_keys=df_keys,
)

O9DataLake.put("StatFcstL1WLag", StatFcstL1WLag)
O9DataLake.put("StatFcstL1MLag", StatFcstL1MLag)
O9DataLake.put("StatFcstL1PMLag", StatFcstL1PMLag)
O9DataLake.put("StatFcstL1QLag", StatFcstL1QLag)
O9DataLake.put("StatFcstL1PQLag", StatFcstL1PQLag)
