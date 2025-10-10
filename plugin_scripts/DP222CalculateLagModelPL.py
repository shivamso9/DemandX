"""
Plugin : DP222CalculateLagModelPL
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    PlanningGrains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]
    LagWindow - All
    NBucketsinMonths - 12

Input Queries:
    StatFcstPL : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Partial Week] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Forecast Iteration].[Forecast Iteration] * [PnL].[Planning PnL] * [Region].[Planning Region] ) on row,  ({Measure.[Stat Fcst PL]}) on column;

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] *[Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {[Time].[Quarter], Key} {[Time].[Planning Quarter], Key};

    SelectedPlanningCycle : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    PlanningCycles : select([Planning Cycle].[Planning Cycle Date]) on row, () on column include memberproperties{[Planning Cycle].[Planning Cycle Date],Key};

Output Variables:
    StatFcstPLLC
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

from helpers.DP222CalculateLagModelPL import main

logger = logging.getLogger("o9_logger")


# Function Calls
StatFcstPL = O9DataLake.get("StatFcstPL")
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
    StatFcstPLLC,
    StatFcstPLWLag,
    StatFcstPLMLag,
    StatFcstPLPMLag,
    StatFcstPLQLag,
    StatFcstPLPQLag,
) = main(
    StatFcstPL=StatFcstPL,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    SelectedPlanningCycle=SelectedPlanningCycle,
    PlanningCycles=PlanningCycles,
    TimeDimension=TimeDimension,
    PlanningGrains=PlanningGrains,
    LagWindow=LagWindow,
    NBucketsinMonths=NBucketsinMonths,
    df_keys=df_keys,
)


O9DataLake.put("StatFcstPLLC", StatFcstPLLC)
O9DataLake.put("StatFcstPLWLag", StatFcstPLWLag)
O9DataLake.put("StatFcstPLMLag", StatFcstPLMLag)
O9DataLake.put("StatFcstPLPMLag", StatFcstPLPMLag)
O9DataLake.put("StatFcstPLQLag", StatFcstPLQLag)
O9DataLake.put("StatFcstPLPQLag", StatFcstPLPQLag)
