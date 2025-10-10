"""
Plugin : DP048PartialWeekLagAssociation
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params: None

Input Queries:
    ForecastTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Forecast Generation Time Bucket], Measure.[Forecast Storage Time Bucket]}) on column;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastParameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Forecast Period]}) on column;

Output Variables:
    PWLagAssociation

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP048PartialWeekLagAssociation import main

logger = logging.getLogger("o9_logger")

# Function Calls
ForecastTimeBucket = O9DataLake.get("ForecastTimeBucket")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
TimeDimension = O9DataLake.get("TimeDimension")
ForecastParameters = O9DataLake.get("ForecastParameters")
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

PWLagAssociation = main(
    ForecastTimeBucket=ForecastTimeBucket,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    ForecastParameters=ForecastParameters,
    df_keys=df_keys,
)

O9DataLake.put("PWLagAssociation", PWLagAssociation)
