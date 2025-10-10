"""
Plugin : DP009OutlierCleansing
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Stat Item],Location.[Stat Location],Account.[Stat Account], Channel.[Stat Channel], Region.[Stat Region], PnL.[Stat PnL], Demand Domain.[Stat Demand Domain]
    ReadFromHive - False
    smooth_fraction - 0.25
    MultiprocessingNumCores - 4

Input Queries:
    Actual : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] *[Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location]
* [Version].[Version Name] * [Item].[Stat Item] * [Time].[Partial Week] ) on row, ({Measure.[Stat Actual], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Stat Actual]), (Measure.[Slice Association Stat] == 1)};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    HistoryPeriod : select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name]) on row, ({Measure.[History Measure], Measure.[History Period]}) on column;

    OutlierParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] *  [Item].[Stat Item] ) on row,  ({Measure.[Outlier Correction], Measure.[Outlier Lower Threshold Limit], Measure.[Outlier Upper Threshold Limit], Measure.[Outlier Method], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Outlier Lower Threshold Limit]), (Measure.[Slice Association Stat] == 1)};

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

	OutlierAbsoluteThreshold : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Outlier Absolute Threshold]}) on column;

	OutlierPercentageThreshold : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Outlier Percentage Threshold]}) on column;

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

Output Variables:
    CleansedData
    ActualL1_output

Slice Dimension Attributes:
    [Sequence].[Sequence]

Pseudo Code:
    - Import the libraries needed for development
    - Define variables to be used in the plugin
    - Derive the level at which one wants to run the outlier process like Item or Item-Location etc.
    - If there is no history the gracefully exit
    - Check if the data source is LS or Hive
    - Filter the Actual dataframe to keep only the relevant columns
    - Handle the options of forecast generation bucket like Week or Month or Planning Month or Quarter or Planning Quarter
    - Get the relevent time columns
    - Add the forecast generation bucket to the Actual data frame
    - Aggregate the Actual to the relevant level
    - Get the history buckets for which we need to perform outlier correction
    - Replace missing values with zeros in relevant history buckets
    - Check if the entire history has only 1 value, if yes, output the same value without any processing
    - If not, get the parameters like upper and lower bound, replace by
    - If the data is less than two cycle and outlier method is seasonal IQR, replace it with rolling sigma
    - Calculate the upper and lower bounds
    - Identify which all points are outliers
    - Replace the outlier by the corrected value
    - Disaggregate the output to partial week level
"""
import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP009OutlierCleansing import main

logger = logging.getLogger("o9_logger")

# Function Calls
Actual = O9DataLake.get("Actual")
StatActual = O9DataLake.get("StatActual")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
HistoryPeriod = O9DataLake.get("HistoryPeriod")
OutlierParameters = O9DataLake.get("OutlierParameters")
OutlierAbsoluteThreshold = O9DataLake.get("OutlierAbsoluteThreshold")
OutlierPercentageThreshold = O9DataLake.get("OutlierPercentageThreshold")
TimeDimension = O9DataLake.get("TimeDimension")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
SellOutOffset = O9DataLake.get("SellOutOffset")

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

CleansedData, ActualL1_output = main(
    Grains=Grains,
    ReadFromHive=ReadFromHive,
    Actual=Actual,
    CurrentTimePeriod=CurrentTimePeriod,
    HistoryPeriod=HistoryPeriod,
    OutlierParameters=OutlierParameters,
    OutlierAbsoluteThreshold=OutlierAbsoluteThreshold,
    OutlierPercentageThreshold=OutlierPercentageThreshold,
    TimeDimension=TimeDimension,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    StatBucketWeight=StatBucketWeight,
    SellOutOffset = SellOutOffset,
    smooth_fraction=float(smooth_fraction),
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    df_keys=df_keys,
)
O9DataLake.put("CleansedData", CleansedData)
O9DataLake.put("ActualL1_output", ActualL1_output)