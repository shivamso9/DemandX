"""
Plugin : DP213CollabRollOverLagAccuracy
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    PlanningGrains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]

Input Queries:
    PlanningActual : Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] ) on row, ({Measure.[Actual L0]}) on column;

    ConsensusFcstWLag : Select ([Planning Cycle].[Planning Cycle Date] * [Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Time].[Week] * [Lag].[Lag] * [Location].[Planning Location] ) on row, ({Measure.[Consensus Fcst W Lag]}) on column;

    PublishedFcstWLag : Select ([Planning Cycle].[Planning Cycle Date] * [Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Time].[Week] * [Lag].[Lag] * [Location].[Planning Location] ) on row, ({Measure.[Published Fcst W Lag]}) on column;

    ConsensusFcstMLag : Select ([Planning Cycle].[Planning Cycle Date] * [Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Time].[Month] * [Lag].[Lag] * [Location].[Planning Location] ) on row, ({Measure.[Consensus Fcst M Lag]}) on column;

    PublishedFcstMLag : Select ([Planning Cycle].[Planning Cycle Date] * [Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Time].[Month] * [Lag].[Lag] * [Location].[Planning Location] ) on row, ({Measure.[Published Fcst M Lag]}) on column;

    ConsensusFcstPMLag : Select ([Planning Cycle].[Planning Cycle Date] * [Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Time].[Planning Month] * [Lag].[Lag] * [Location].[Planning Location] ) on row, ({Measure.[Consensus Fcst PM Lag]}) on column;

    PublishedFcstPMLag : Select ([Planning Cycle].[Planning Cycle Date] * [Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Time].[Planning Month] * [Lag].[Lag] * [Location].[Planning Location] ) on row, ({Measure.[Published Fcst PM Lag]}) on column;


    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key};

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} ;

Output Variables:
    ConsensusFcstWLagAbsError
    PublishedFcstWLagAbsError
    ConsensusFcstMLagAbsError
    PublishedFcstMLagAbsError
    ConsensusFcstPMLagAbsError
    PublishedFcstPMLagAbsError
    LastMonthsAccuracy


Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP213CollabRollOverLagAccuracy import main

logger = logging.getLogger("o9_logger")

# Function Calls
PlanningActual = O9DataLake.get("PlanningActual")
ConsensusFcstWLag = O9DataLake.get("ConsensusFcstWLag")
ConsensusFcstMLag = O9DataLake.get("ConsensusFcstMLag")
ConsensusFcstPMLag = O9DataLake.get("ConsensusFcstPMLag")
PublishedFcstWLag = O9DataLake.get("PublishedFcstWLag")
PublishedFcstMLag = O9DataLake.get("PublishedFcstMLag")
PublishedFcstPMLag = O9DataLake.get("PublishedFcstPMLag")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
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
    ConsensusFcstWLagAbsError,
    PublishedFcstWLagAbsError,
    ConsensusFcstMLagAbsError,
    PublishedFcstMLagAbsError,
    ConsensusFcstPMLagAbsError,
    PublishedFcstPMLagAbsError,
    LastMonthsAccuracy,
) = main(
    PlanningActual=PlanningActual,
    ConsensusFcstWLag=ConsensusFcstWLag,
    ConsensusFcstMLag=ConsensusFcstMLag,
    ConsensusFcstPMLag=ConsensusFcstPMLag,
    PublishedFcstWLag=PublishedFcstWLag,
    PublishedFcstMLag=PublishedFcstMLag,
    PublishedFcstPMLag=PublishedFcstPMLag,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    PlanningGrains=PlanningGrains,
    TimeLevel=TimeLevel,
    df_keys=df_keys,
)

O9DataLake.put("ConsensusFcstWLagAbsError", ConsensusFcstWLagAbsError)
O9DataLake.put("PublishedFcstWLagAbsError", PublishedFcstWLagAbsError)
O9DataLake.put("ConsensusFcstMLagAbsError", ConsensusFcstMLagAbsError)
O9DataLake.put("PublishedFcstMLagAbsError", PublishedFcstMLagAbsError)
O9DataLake.put("ConsensusFcstPMLagAbsError", ConsensusFcstPMLagAbsError)
O9DataLake.put("PublishedFcstPMLagAbsError", PublishedFcstPMLagAbsError)
O9DataLake.put("LastMonthsAccuracy", LastMonthsAccuracy)
