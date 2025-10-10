"""
    Plugin : DP204SystemStatRetail
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    HistoryMeasure - Actual Cleansed Sliced
    UseHolidays - False
    IncludeDiscIntersections - True
    AlgoListMeasure - Assigned Algorithm List
    MultiprocessingNumCores - 4
    PlanningGrains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]

Input Queries:
    AlgoList : Select ([Sequence].[Sequence] * [Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] ) on row,  ({Measure.[Assigned Algorithm List], Measure.[Bestfit Algorithm LC], Measure.[Bestfit Algorithm], Measure.[Stat Fcst L1 Flag], Measure.[Trend L1], Measure.[Seasonality L1], Measure.[Planner Bestfit Algorithm], Measure.[Assigned Rule], Measure.[Planner Assigned Algorithm List], Measure.[System Assigned Algorithm List], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Assigned Algorithm List]), (Measure.[Slice Association Stat] == 1)};

    Actual : 175 Actual Cleansed Sliced (Actual Cleansed Sliced)

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

    ForecastParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row,  ({  Measure.[History Period], Measure.[Forecast Period], Measure.[Validation Period], Measure.[Bestfit Method], Measure.[Error Metric], Measure.[History Time Buckets]}) on column;

    AlgoParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Stat Parameter].[Stat Parameter] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * [Stat Algorithm].[Stat Algorithm] * [Sequence].[Sequence] ) on row,  ({Measure.[System Stat Parameter Value], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[System Stat Parameter Value]), (Measure.[Slice Association Stat] == 1)};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

    HolidayData : Select ([Version].[Version Name] * [Region].[Planning Region] * [Time].[Day] ) on row, ({Measure.[Holiday Type]}) on column;

    StatRegionMapping : Select ([Region].[Planning Region] * [Region].[Stat Region]) on row, () on column;

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    MasterAlgoList : Select ([Version].[Version Name]) on row, ({Measure.[Assigned Algorithm List]}) on column;

    DefaultAlgoParameters : Select ([Version].[Version Name] * [Stat Algorithm].[Stat Algorithm] * [Stat Parameter].[Stat Parameter] ) on row, ({Measure.[Stat Algorithm Parameter Association]}) on column include memberproperties {[Stat Parameter].[Stat Parameter], [Stat Parameter Weekly Default]} {[Stat Parameter].[Stat Parameter], [Stat Parameter Monthly Default]};

Output Variables:
    All are deltalake measure groups:
        SystemStatFcstL1 : 190 Stat System L1 (all candidate forecasts + bounds + system stat fcst l1)
        StatFcstL1 : 192 Stat L1 (Stat Fcst L1)
        ModelParameters : 194 Stat Models L1 (Validation Error, Validation Method, Run Time, Algorithm Parameters)
        SystemBestfitAlgorithm : 196 Bestfit Algorithm System L1 (System Bestfit Algorithm)
        BestfitAlgorithm : 198 Bestfit Algorithm L1 (Bestfit Algorithm)
        StatFcstPL : 204 Stat System PL (Stat Fcst PL)

Slice Dimension Attributes:
    Sequence.[Sequence]

Pseudocode :

* If the actuals is empty/sum of actuals is zero for a particular slice, exit with fallback dataframe
* Replace any negative values in actuals with zero
* Infer the parameter value for ‘Forecast Generation Time Bucket’ and aggregate the actuals from Partial Week level to Week/Month/Planning Month
* Get the last n time periods based on the ‘History Period’ parameter and filter data for the last n time periods
* If actuals is empty/sum of actuals is zero for last n periods, exit with fallback dataframe
* Exclude the discontinued intersections from AlgoList according to the script parameter
* Merge algolist with actuals dataframe to add the algorithm column to the actuals dataframe
* Depending on the number of minimum points set inside the code, filter out intersections which do not have the min data points constraint satisfied
* Assign parameters to all the forecasting algorithms assigned
* * If AlgoParameters input is empty, assign default parameters to all the intersections
* * If AlgoParameters input is not empty, assign default parameters to only the intersections for which parameters are not assigned in input
* Pre process Holiday data
* * Aggregate it to the Stat Customer Group and relevant time bucket (Week/Month/Planning Month)
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

from helpers.DP204SystemStatRetail import main

logger = logging.getLogger("o9_logger")


# Function Calls
AlgoList = O9DataLake.get("AlgoList")
Actual = O9DataLake.get("Actual")
TimeDimension = O9DataLake.get("TimeDimension")
ForecastParameters = O9DataLake.get("ForecastParameters")
AlgoParameters = O9DataLake.get("AlgoParameters")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
HolidayData = O9DataLake.get("HolidayData")
StatRegionMapping = O9DataLake.get("StatRegionMapping")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
MasterAlgoList = O9DataLake.get("MasterAlgoList")
DefaultAlgoParameters = O9DataLake.get("DefaultAlgoParameters")

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
    SystemStatFcstL1,
    StatFcstL1,
    ModelParameters,
    SystemBestfitAlgorithm,
    BestfitAlgorithm,
    StatFcstPL,
) = main(
    Grains=Grains,
    history_measure=HistoryMeasure,
    AlgoList=AlgoList,
    Actual=Actual,
    TimeDimension=TimeDimension,
    ForecastParameters=ForecastParameters,
    AlgoParameters=AlgoParameters,
    CurrentTimePeriod=CurrentTimePeriod,
    HolidayData=HolidayData,
    StatRegionMapping=StatRegionMapping,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    StatBucketWeight=StatBucketWeight,
    MasterAlgoList=MasterAlgoList,
    DefaultAlgoParameters=DefaultAlgoParameters,
    UseHolidays=UseHolidays,
    IncludeDiscIntersections=IncludeDiscIntersections,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    AlgoListMeasure=AlgoListMeasure,
    df_keys=df_keys,
    PlanningGrains=PlanningGrains,
)

O9DataLake.put("SystemStatFcstL1", SystemStatFcstL1)
O9DataLake.put("StatFcstL1", StatFcstL1)
O9DataLake.put("ModelParameters", ModelParameters)
O9DataLake.put("SystemBestfitAlgorithm", SystemBestfitAlgorithm)
O9DataLake.put("BestfitAlgorithm", BestfitAlgorithm)
O9DataLake.put("StatFcstPL", StatFcstPL)
