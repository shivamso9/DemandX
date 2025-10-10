"""
    Plugin : DP015SystemStat
    Version : 2025.08.00
    Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    HistoryMeasure - Actual Cleansed
    IncludeDiscIntersections - True
    AlgoListMeasure - Assigned Algorithm List
    MultiprocessingNumCores - 4

Input Queries:
    AlgoList : Select ([Sequence].[Sequence] * [Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] ) on row,  ({Measure.[Assigned Algorithm List], Measure.[Bestfit Algorithm LC], Measure.[Bestfit Algorithm], Measure.[Bestfit Algorithm LC Final], Measure.[Stat Fcst L1 Flag], Measure.[Trend L1], Measure.[Seasonality L1], Measure.[Planner Bestfit Algorithm], Measure.[Assigned Rule], Measure.[Planner Assigned Algorithm List], Measure.[System Assigned Algorithm List], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Assigned Algorithm List]), (Measure.[Slice Association Stat] == 1)};

    Actual : Select ([Sequence].[Sequence] *[Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * [Time].[Partial Week] ) on row,  ({Measure.[Actual Cleansed], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Actual Cleansed]), (Measure.[Slice Association Stat] == 1)};

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter] * [Time].[Week Name]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row,  ({  Measure.[History Period], Measure.[Forecast Period], Measure.[Validation Period], Measure.[Bestfit Method], Measure.[Error Metric], Measure.[History Time Buckets]  }) on column;

    AlgoParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Stat Parameter].[Stat Parameter] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * [Stat Algorithm].[Stat Algorithm] * [Sequence].[Sequence] ) on row,  ({Measure.[System Stat Parameter Value], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[System Stat Parameter Value]), (Measure.[Slice Association Stat] == 1)};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    StatSegment : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Class].[Class] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Item].[Stat Item] * [Sequence].[Sequence] ) on row,  ({Measure.[Product Customer L1 Segment], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Product Customer L1 Segment]), (Measure.[Slice Association Stat] == 1)};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    MasterAlgoList : Select ([Version].[Version Name]) on row, ({Measure.[Assigned Algorithm List]}) on column;

    DefaultAlgoParameters : Select ([Version].[Version Name] * [Stat Algorithm].[Stat Algorithm] * [Stat Parameter].[Stat Parameter] ) on row, ({Measure.[Stat Algorithm Parameter Association]}) on column include memberproperties {[Stat Parameter].[Stat Parameter], [Stat Parameter Weekly Default]} {[Stat Parameter].[Stat Parameter], [Stat Parameter Monthly Default]} {[Stat Parameter].[Stat Parameter], [Stat Parameter Quarterly Default]};

    SeasonalIndices :  Select ([Sequence].[Sequence] * [Forecast Iteration].[Forecast Iteration] * [Channel].[Stat Channel] * [Version].[Version Name] * [Account].[Stat Account] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Region].[Stat Region] * [Location].[Stat Location]  * [Item].[Stat Item] * [Time].[Partial Week]) on row, ({Measure.[SCHM Validation Seasonal Index],Measure.[SCHM Seasonal Index], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[SCHM Seasonal Index]), (Measure.[Slice Association Stat] == 1)};

SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;


Output Variables:
    AllForecast
    ForecastModel

Slice Dimension Attributes:
    Sequence.[Sequence]

Pseudocode :

* If the actuals is empty/sum of actuals is zero for a particular slice, exit with fallback dataframe
* Replace any negative values in actuals with zero
* Infer the parameter value for ‘Forecast Generation Time Bucket’ and aggregate the actuals from Partial Week level to Week/Month/Planning Month/Quarter/Planning Quarter
* Get the last n time periods based on the ‘History Period’ parameter and filter data for the last n time periods
* If actuals is empty/sum of actuals is zero for last n periods, exit with fallback dataframe
* Exclude the discontinued intersections from AlgoList according to the script parameter
* Merge algolist with actuals dataframe to add the algorithm column to the actuals dataframe
* Depending on the number of minimum points set inside the code, filter out intersections which do not have the min data points constraint satisfied
* Assign parameters to all the forecasting algorithms assigned
* * If AlgoParameters input is empty, assign default parameters to all the intersections
* * If AlgoParameters input is not empty, assign default parameters to only the intersections for which parameters are not assigned in input
* Pre process Holiday data
* * Aggregate it to the Stat Customer Group and relevant time bucket (Week/Month/Planning Month/Quarter/Planning Quarter)
* Sort the actuals dataframe by intersection and time so that we have the time series in the correct order
* Run function for every intersection (using joblib multiprocessing) and generate the forecast
* * Collect list of algorithms to be evaluated for every intersection
* * Initialize the AlgoParamExtractor class to fetch parameters for every algorithm
* * Set the validation method to in sample/out sample after evaluating the dataset size
* * Pass 1 : Generate forecasts (using fit_models) for the assigned algorithms for the validation horizon (it could be in sample/out sample)
* * * Generate the forecasts including forecast bounds
* * * Create the model description string by deriving the fitted model parameters
* * Filter the number of datapoints as specified in ‘Validation Periods’ measure
* Pass 2 : Generate forecasts  (using fit_models) for the assigned algorithms for the future time periods

* Validate the forecast output
* Check if forecast has been generated against all combinations and generate a warning for the missing ones
* Disaggregate forecast numbers to Partial Week level
* Columns for all algorithms should be present in both output dataframes, check and add the missing ones
* AllForecast - forecast numbers generated at intersection level
* ForecastModel - forecast model descriptions which has the fitted model parameters
* Output the results to o9DataLake

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP015SystemStat import main

logger = logging.getLogger("o9_logger")


# Function Calls
AlgoList = O9DataLake.get("AlgoList")
Actual = O9DataLake.get("Actual")
TimeDimension = O9DataLake.get("TimeDimension")
ForecastParameters = O9DataLake.get("ForecastParameters")
AlgoParameters = O9DataLake.get("AlgoParameters")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
StatSegment = O9DataLake.get("StatSegment")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
MasterAlgoList = O9DataLake.get("MasterAlgoList")
DefaultAlgoParameters = O9DataLake.get("DefaultAlgoParameters")
SeasonalIndices = O9DataLake.get("SeasonalIndices")
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

AllForecast, ForecastModel = main(
    Grains=Grains,
    history_measure=HistoryMeasure,
    AlgoList=AlgoList,
    Actual=Actual,
    TimeDimension=TimeDimension,
    ForecastParameters=ForecastParameters,
    AlgoParameters=AlgoParameters,
    CurrentTimePeriod=CurrentTimePeriod,
    non_disc_intersections=StatSegment,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    StatBucketWeight=StatBucketWeight,
    MasterAlgoList=MasterAlgoList,
    DefaultAlgoParameters=DefaultAlgoParameters,
    SeasonalIndices=SeasonalIndices,
    IncludeDiscIntersections=IncludeDiscIntersections,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    AlgoListMeasure=AlgoListMeasure,
    df_keys=df_keys,
    SellOutOffset=SellOutOffset,
    model_params=eval(model_params),
)
O9DataLake.put("AllForecast", AllForecast)
O9DataLake.put("ForecastModel", ForecastModel)
