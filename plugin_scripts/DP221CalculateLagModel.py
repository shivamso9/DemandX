"""
Plugin : DP221CalculateLagModel
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    PlanningGrains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]
    TimeLevel - Week
    LagWindow - All

Input Queries:
    StatFcst : Select ([Version].[Version Name] * [Time].[Partial Week] * [Account].[Planning Account] * [Item].[Planning Item] * [Location].[Planning Location] * [Region].[Planning Region] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] ) on row, ({Measure.[Stat Fcst]}) on column;

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] *[Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {[Time].[Quarter], Key} {[Time].[Planning Quarter], Key};

    SelectedPlanningCycle : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] *[Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {[Time].[Quarter], Key} {[Time].[Planning Quarter], Key};

    PlanningCycles : select([Planning Cycle].[Planning Cycle Date]) on row, () on column include memberproperties{[Planning Cycle].[Planning Cycle Date],Key};

Output Variables:
    StatFcstPLWLag
    StatFcstPLMLag
    StatFcstPLPMLag
    StatFcstPLQLag
    StatFcstPLPQLag


Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP221CalculateLagModel import main

logger = logging.getLogger("o9_logger")


# Function Calls
StatFcst = O9DataLake.get("StatFcst")
SelectedPlanningCycle = O9DataLake.get("SelectedPlanningCycle")
PlanningCycles = O9DataLake.get("PlanningCycles")
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
    StatFcstPLWLag,
    StatFcstPLMLag,
    StatFcstPLPMLag,
    StatFcstPLQLag,
    StatFcstPLPQLag,
) = main(
    StatFcst=StatFcst,
    SelectedPlanningCycle=SelectedPlanningCycle,
    PlanningCycles=PlanningCycles,
    TimeDimension=TimeDimension,
    PlanningGrains=PlanningGrains,
    LagWindow=LagWindow,
    TimeLevel=TimeLevel,
    df_keys=df_keys,
)

O9DataLake.put("StatFcstPLWLag", StatFcstPLWLag)
O9DataLake.put("StatFcstPLMLag", StatFcstPLMLag)
O9DataLake.put("StatFcstPLPMLag", StatFcstPLPMLag)
O9DataLake.put("StatFcstPLQLag", StatFcstPLQLag)
O9DataLake.put("StatFcstPLPQLag", StatFcstPLPQLag)
