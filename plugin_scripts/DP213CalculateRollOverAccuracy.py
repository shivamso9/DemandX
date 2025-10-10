"""
Plugin : DP213CalculateRollOverAccuracy
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    PlanningGrains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]
    TimeLevel - Week
    AccuracyWindow - 6


Input Queries:
    PlanningActual : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row, ({Measure.[Actual]}) on column;

    StatFcstPLWLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Week] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst PL W Lag]}) on column include memberproperties{[Planning Cycle].[Planning Cycle Date], Key };

    StatFcstPLMLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Month] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst PL M Lag]}) on column include memberproperties{[Planning Cycle].[Planning Cycle Date], Key };

    StatFcstPLPMLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Planning Month] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst PL PM Lag]}) on column include memberproperties{[Planning Cycle].[Planning Cycle Date], Key };

    StatFcstPLQLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Quarter] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst PL Q Lag]}) on column include memberproperties{[Planning Cycle].[Planning Cycle Date], Key };

    StatFcstPLPQLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Planning Quarter] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,  ({Measure.[Stat Fcst PL PQ Lag]}) on column include memberproperties{[Planning Cycle].[Planning Cycle Date], Key };

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;


Output Variables:
    StatFcstAbsError
    StatFcstPLWLagAbsError
    StatFcstPLMLagAbsError
    StatFcstPLPMLagAbsError
    StatFcstPLQLagAbsError
    StatFcstPLPQLagAbsError

Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP213CalculateRollOverAccuracy import main

logger = logging.getLogger("o9_logger")


# Function Calls
PlanningActual = O9DataLake.get("PlanningActual")
StatFcstPLWLag = O9DataLake.get("StatFcstPLWLag")
StatFcstPLMLag = O9DataLake.get("StatFcstPLMLag")
StatFcstPLPMLag = O9DataLake.get("StatFcstPLPMLag")
StatFcstPLQLag = O9DataLake.get("StatFcstPLQLag")
StatFcstPLPQLag = O9DataLake.get("StatFcstPLPQLag")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
TimeDimension = O9DataLake.get("TimeDimension")
StatBucketWeight = O9DataLake.get("StatBucketWeight")

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
    StatFcstAbsError,
    StatFcstPLWLagAbsError,
    StatFcstPLMLagAbsError,
    StatFcstPLPMLagAbsError,
    StatFcstPLQLagAbsError,
    StatFcstPLPQLagAbsError,
) = main(
    PlanningActual=PlanningActual,
    StatFcstPLWLag=StatFcstPLWLag,
    StatFcstPLMLag=StatFcstPLMLag,
    StatFcstPLPMLag=StatFcstPLPMLag,
    StatFcstPLQLag=StatFcstPLQLag,
    StatFcstPLPQLag=StatFcstPLPQLag,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    PlanningGrains=PlanningGrains,
    StatBucketWeight=StatBucketWeight,
    AccuracyWindow=AccuracyWindow,
    TimeLevel=TimeLevel,
    df_keys=df_keys,
)

O9DataLake.put("StatFcstAbsError", StatFcstAbsError)
O9DataLake.put("StatFcstPLWLagAbsError", StatFcstPLWLagAbsError)
O9DataLake.put("StatFcstPLMLagAbsError", StatFcstPLMLagAbsError)
O9DataLake.put("StatFcstPLPMLagAbsError", StatFcstPLPMLagAbsError)
O9DataLake.put("StatFcstPLQLagAbsError", StatFcstPLQLagAbsError)
O9DataLake.put("StatFcstPLPQLagAbsError", StatFcstPLPQLagAbsError)
