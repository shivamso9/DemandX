"""
Plugin : DP050PopulatePLCHorizon
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:

Input Queries:
    # TODO : Combine both Intro and Disco measures in one IBPL input query
    IntroDate - Select ([Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] *  [Location].[Planning Location]  * [Version].[Version Name] * [Item].[Planning Item] ) on row, ({Measure.[Intro Date]}) on column;

    DiscoDate - Select ([Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] *  [Location].[Planning Location]  * [Version].[Version Name] * [Item].[Planning Item] ) on row, ({Measure.[Disco Date]}) on column;

    TimeDimension - select (&AllMonths * &AllPlanningMonths * &AllWeeks * &AllPartialWeeks) include memberproperties {[Time].[Week], Key}{[Time].[Month], Key} {[Time].[Planning Month], Key} {[Time].[Partial Week], Key};

    FcstStorageTimeBucket - Select ([Version].[Version Name] ) on row, ({Measure.[Forecast Storage Time Bucket]}) on column;

    ConsensusFcstBuckets - select (&ConsensusForecastBuckets);

    # TODO : Remove this, use TimeDimension instead
    PlanningMonthDim - Select (&AllPlanningMonths * &AllWeeks * &AllPartialWeeks) on row, () on column include memberproperties {[Time].[Week], Key} {[Time].[Planning Month], Key} {[Time].[Partial Week], Key};

    # TODO : Remove this, use TimeDimension instead
    MonthDim - select (&AllMonths * &AllWeeks * &AllPartialWeeks) on row, () on column include memberproperties {[Time].[Week], Key}{[Time].[Month], Key} {[Time].[Partial Week], Key};

Output Variables:
    Output

Slice Dimension Attributes:

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP050PopulatePLCHorizon import main

logger = logging.getLogger("o9_logger")

# Function Calls
IntroDate = O9DataLake.get("IntroDate")
DiscoDate = O9DataLake.get("DiscoDate")
MonthDim = O9DataLake.get("MonthDim")
PlanningMonthDim = O9DataLake.get("PlanningMonthDim")
FcstStorageTimeBucket = O9DataLake.get("FcstStorageTimeBucket")
ConsensusFcstBuckets = O9DataLake.get("ConsensusFcstBuckets")
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

Output = main(
    IntroDate=IntroDate,
    DiscoDate=DiscoDate,
    MonthDim=MonthDim,
    PlanningMonthDim=PlanningMonthDim,
    FcstStorageTimeBucket=FcstStorageTimeBucket,
    ConsensusFcstBuckets=ConsensusFcstBuckets,
    df_keys=df_keys,
)

O9DataLake.put("Output", Output)
