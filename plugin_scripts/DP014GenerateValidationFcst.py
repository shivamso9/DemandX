"""
Plugin : DP014GenerateValidationFcst
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:
    ACFLowerThreshold - -0.10
    ACFUpperThreshold - 0.90
    ACFSkipLags - 11
    ACFDiff - 0
    AlphaACF - 0.05
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    IncludeDiscIntersections - True
    LagsToStore - All
    RequiredACFLagsInWeeks - 12,11,6,26,52
    ReadFromHive - False
    RUN_VALIDATION_EVERY_FOLD - True
    smooth_fraction - 0.25
    MultiprocessingNumCores - 4
    PlannerOverrideCycles - None
    TrendThreshold - 30

Input Queries:
    StatLevelActual : Select ([Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Segmentation LOB] * [Item].[Stat Item] * [Time].[Partial Week] ) on row, ({Measure.[Stat Actual]}) on column;

    Parameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Bestfit Method], Measure.[Disco Period], Measure.[Error Metric], Measure.[Forecast Period], Measure.[History Measure], Measure.[History Period], Measure.[History Time Buckets], Measure.[Intermittency Threshold], Measure.[New Launch Period], Measure.[Seasonality Threshold], Measure.[Trend Threshold], Measure.[Validation Period], Measure.[Volume-COV History Period], Measure.[Forecast Strategy]}) on column;

    OutlierParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] ) on row, ({Measure.[Outlier Correction], Measure.[Outlier Upper Threshold Limit], Measure.[Outlier Lower Threshold Limit], Measure.[Outlier Method]}) on column;

    AlgoParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Stat Parameter].[Stat Parameter] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * [Stat Algorithm].[Stat Algorithm] ) on row, ({Measure.[System Stat Parameter Value]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter] *[Time].[Week Name]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    AlgoList: Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Location].[Stat Location] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Item].[Stat Item] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row,
({Measure.[Assigned Algorithm List]}) on column;

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    MasterAlgoList : Select ([Version].[Version Name]) on row, ({Measure.[Assigned Algorithm List]}) on column;

    DefaultAlgoParameters : Select ([Version].[Version Name] * [Stat Algorithm].[Stat Algorithm] * [Stat Parameter].[Stat Parameter] ) on row, ({Measure.[Stat Algorithm Parameter Association]}) on column include memberproperties {[Stat Parameter].[Stat Parameter], [Stat Parameter Weekly Default]} {[Stat Parameter].[Stat Parameter], [Stat Parameter Monthly Default]} {[Stat Parameter].[Stat Parameter], [Stat Parameter Quarterly Default]};

    BestFitSelectionCriteria : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row, ({Measure.[Bestfit Selection Criteria]}) on column;

    CurrentCycleBestfitAlgorithm : Select ([Version].[Version Name] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] ) on row, ({Measure.[Bestfit Algorithm]}) on column;

    SeasonalIndices :  Select ([Sequence].[Sequence] * [Forecast Iteration].[Forecast Iteration] * [Channel].[Stat Channel] * [Version].[Version Name] * [Account].[Stat Account] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Region].[Stat Region] * [Location].[Stat Location]  * [Item].[Stat Item] * [Time].[Partial Week]) on row, ({Measure.[SCHM Validation Seasonal Index Backtest],Measure.[SCHM Seasonal Index Backtest], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[SCHM Seasonal Index Backtest]), (Measure.[Slice Association Stat] == 1)};

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

    StatSegmentation : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Planning Cycle].[Planning Cycle Date] * [Region].[Stat Region] * [Location].[Stat Location] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Item].[Stat Item] * [Account].[Stat Account]) on row, ({Measure.[Volume Segment L1 Planning Cycle], Measure.[COV Segment L1 Planning Cycle], Measure.[Product Segment L1 Planning Cycle], Measure.[Length of Series L1 Planning Cycle], Measure.[Number of Zeros L1 Planning Cycle], Measure.[Intermittent L1 Planning Cycle], Measure.[PLC Status L1 Planning Cycle], Measure.[Std Dev L1 Planning Cycle], Measure.[Avg Volume L1 Planning Cycle], Measure.[COV L1 Planning Cycle], Measure.[Volume L1 Planning Cycle], Measure.[Volume % L1 Planning Cycle], Measure.[Cumulative Volume % L1 Planning Cycle], Measure.[Stat Disc Date System L1 Planning Cycle], Measure.[Stat Intro Date System L1 Planning Cycle]}) on column;

    ProductSegmentation : Select ([Version].[Version Name] * [Item].[Stat Item] * [Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Class].[Class] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Planning Cycle].[Planning Cycle Date]) on row, ({Measure.[Product Customer L1 Segment Planning Cycle]}) on column;

    PlanningCycleDates : Select ([Planning Cycle].[Planning Cycle Date]) on row, ({Measure.[PlanningCycleDateKey]}) on column;

    ValidationParameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Validation Fold], Measure.[Validation Step Size], Measure.[Validation Period]}) on column;

    ForecastEngine : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Forecast Engine]}) on column;

Output Variables:
    AllForecastWithPC

Slice Dimension Attributes:
    Sequence.[Sequence]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP014GenerateValidationFcst import main

logger = logging.getLogger("o9_logger")


# Function Calls
StatLevelActual = O9DataLake.get("StatLevelActual")
Parameters = O9DataLake.get("Parameters")
OutlierParameters = O9DataLake.get("OutlierParameters")
AlgoParameters = O9DataLake.get("AlgoParameters")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
AlgoList = O9DataLake.get("AlgoList")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
MasterAlgoList = O9DataLake.get("MasterAlgoList")
DefaultAlgoParameters = O9DataLake.get("DefaultAlgoParameters")
BestFitSelectionCriteria = O9DataLake.get("BestFitSelectionCriteria")
CurrentCycleBestfitAlgorithm = O9DataLake.get("CurrentCycleBestfitAlgorithm")
SeasonalIndices = O9DataLake.get("SeasonalIndices")
SellOutOffset = O9DataLake.get("SellOutOffset")
StatSegmentation = O9DataLake.get("StatSegmentation")
ProductSegmentation = O9DataLake.get("ProductSegmentation")
PlanningCycleDates = O9DataLake.get("PlanningCycleDates")
ValidationParameters = O9DataLake.get("ValidationParameters")
ForecastEngine = O9DataLake.get("ForecastEngine")

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
AllForecastWithPC = main(
    ACFLowerThreshold=ACFLowerThreshold,
    ACFUpperThreshold=ACFUpperThreshold,
    ACFSkipLags=ACFSkipLags,
    ACFDiff=ACFDiff,
    alpha=AlphaACF,
    Grains=Grains,
    IncludeDiscIntersections=IncludeDiscIntersections,
    LagsToStore=LagsToStore,
    RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
    ReadFromHive=ReadFromHive,
    RUN_VALIDATION_EVERY_FOLD=eval(RUN_VALIDATION_EVERY_FOLD),
    PlannerOverrideCycles=PlannerOverrideCycles,
    smooth_fraction=smooth_fraction,
    StatLevelActual=StatLevelActual,
    Parameters=Parameters,
    OutlierParameters=OutlierParameters,
    AlgoParameters=AlgoParameters,
    TimeDimension=TimeDimension,
    CurrentTimePeriod=CurrentTimePeriod,
    AlgoList=AlgoList,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    StatBucketWeight=StatBucketWeight,
    MasterAlgoList=MasterAlgoList,
    DefaultAlgoParameters=DefaultAlgoParameters,
    BestFitSelectionCriteria=BestFitSelectionCriteria,
    CurrentCycleBestfitAlgorithm=CurrentCycleBestfitAlgorithm,
    SeasonalIndices=SeasonalIndices,
    SellOutOffset=SellOutOffset,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    StatSegmentation=StatSegmentation,
    PlanningCycleDates=PlanningCycleDates,
    ValidationParameters=ValidationParameters,
    ForecastEngine=ForecastEngine,
    TrendThreshold=TrendThreshold,
    df_keys=df_keys,
)
O9DataLake.put("AllForecastWithPC", AllForecastWithPC)
