"""
Plugin : DP213IBPRollOverLagAccuracy
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    PlanningGrains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]

Input Queries:
    PlanningActual : Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] ) on row, ({Measure.[Actual L0]}) on column;

    OutlookLag1 : Select ([Version].[Version Name] * [Region].[Planning Region] * [Location].[Planning Location] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Time].[Month] * [Account].[Planning Account] ) on row, ({Measure.[Outlook Lag1]}) on column;

    UnconsOutlookLag1 : Select ([Version].[Version Name] * [Region].[Planning Region] * [Location].[Planning Location] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Time].[Month] * [Account].[Planning Account] ) on row, ({Measure.[Uncons Outlook Lag1]}) on column;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key};

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} ;

Output Variables:
    OutlookAbsError
    UnconsOutlookAbsError


Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP213IBPRollOverLagAccuracy import main

logger = logging.getLogger("o9_logger")

# Function Calls
PlanningActual = O9DataLake.get("PlanningActual")
OutlookLag1 = O9DataLake.get("OutlookLag1")
UnconsOutlookLag1 = O9DataLake.get("UnconsOutlookLag1")
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

OutlookAbsError, UnconsOutlookAbsError = main(
    PlanningActual=PlanningActual,
    OutlookLag1=OutlookLag1,
    UnconsOutlookLag1=UnconsOutlookLag1,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    PlanningGrains=PlanningGrains,
    df_keys=df_keys,
)

O9DataLake.put("OutlookAbsError", OutlookAbsError)
O9DataLake.put("UnconsOutlookAbsError", UnconsOutlookAbsError)
