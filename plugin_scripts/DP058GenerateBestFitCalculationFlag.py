"""
Plugin : DP058GenerateBestFitCalculationFlag
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params: None

Input Queries:
    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    BestFitCalcFrequency : Select ([Version].[Version Name] * [Item].[Segmentation LOB]  * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Bestfit Calculation Frequency]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    BestFitFrequencyStartDate : Select ([Version].[Version Name] * [Item].[Segmentation LOB]  *  [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Bestfit Frequency Start Date]}) on column;

    CurrentDay : Select (&CurrentDay) on row, () on column include memberproperties {[Time].[Day], Key};

Output Variables:
    Output

Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP058GenerateBestFitCalculationFlag import main

logger = logging.getLogger("o9_logger")


# Function Calls
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
BestFitCalcFrequency = O9DataLake.get("BestFitCalcFrequency")
TimeDimension = O9DataLake.get("TimeDimension")
BestFitFrequencyStartDate = O9DataLake.get("BestFitFrequencyStartDate")
CurrentDay = O9DataLake.get("CurrentDay")

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
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    BestFitCalcFrequency=BestFitCalcFrequency,
    TimeDimension=TimeDimension,
    BestFitFrequencyStartDate=BestFitFrequencyStartDate,
    CurrentDay=CurrentDay,
    df_keys=df_keys,
)

O9DataLake.put("Output", Output)
