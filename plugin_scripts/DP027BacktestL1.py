"""
Plugin : DP027BacktestL1
Version : 2025.08
Maintained by : dpref@o9solutions.com

Script Params:
    ReadFromHive - False
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    AlphaACF - 0.05
    MultiprocessingNumCores - 4
    ACFLowerThreshold - -0.10
    ACFUpperThreshold - 0.90
    ACFSkipLags - 11
    ACFDiff - 0
    RequiredACFLagsInWeeks - 12,11,6,26,52

    smooth_fraction - 0.25

    UseHolidays - False
    IncludeDiscIntersections - True

    OverrideFlatLineForecasts - False

    TrendVariationThreshold - 0.6
    LevelVariationThreshold - 0.6
    RangeVariationThreshold - 0.6
    SeasonalVariationThreshold - 0.6
    SeasonalVariationCountThreshold - 0.6
    MinimumIndicePercentage - 0.01
    AbsTolerance - 50
    COCCVariationThreshold - 0.25
    ReasonabilityCycles - 1

    RUN_SEGMENTATION_EVERY_CYCLE - True
    RUN_BEST_FIT_EVERY_CYCLE - True
    RUN_VALIDATION_EVERY_FOLD - True
    BackTestCyclePeriod - 2,4,6
    RolloverAlgorithmflag - False
    LagsToStore - All

    BacktestOnlyStatAlgos - False
    BacktestOnlyCustomAlgos - False
    LagNInput - 1
    TrendThreshold - 20

Input Queries:
    StatLevelActual : Select ([Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Segmentation LOB] * [Item].[Stat Item] * [Time].[Partial Week] ) on row, ({Measure.[Stat Actual]}) on column;

    Parameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Bestfit Method], Measure.[Disco Period], Measure.[Error Metric], Measure.[Forecast Period], Measure.[History Measure], Measure.[History Period], Measure.[History Time Buckets], Measure.[Intermittency Threshold], Measure.[New Launch Period], Measure.[Seasonality Threshold], Measure.[Trend Threshold], Measure.[Validation Period], Measure.[Volume-COV History Period], Measure.[Forecast Strategy]}) on column;

    SegmentationParameters : Select ([Class].[Class] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Volume Threshold], Measure.[COV Threshold]}) on column;

    OutlierParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] ) on row, ({Measure.[Outlier Correction], Measure.[Outlier Upper Threshold Limit], Measure.[Outlier Lower Threshold Limit], Measure.[Outlier Method]}) on column;

    AlgoParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Stat Parameter].[Stat Parameter] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * [Stat Algorithm].[Stat Algorithm] ) on row, ({Measure.[System Stat Parameter Value]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter] *[Time].[Week Name]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    Rules : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Stat Rule].[Stat Rule] *{Measure.[Planner Volume Segment],Measure.[Planner COV Segment],Measure.[Planner Intermittency],Measure.[Planner PLC],Measure.[Planner Trend],Measure.[Planner Seasonality],Measure.[Planner Length of Series], Measure.[Planner Algorithm List]} ) on row,  () on column include memberproperties {[Stat Rule].[Stat Rule],[System Rule Description],[System Algorithm List]} where {~(Measure.[Planner Algorithm List] contains "No Forecast")};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    MasterAlgoList : Select ([Version].[Version Name]) on row, ({Measure.[Assigned Algorithm List]}) on column;

    DefaultAlgoParameters : Select ([Version].[Version Name] * [Stat Algorithm].[Stat Algorithm] * [Stat Parameter].[Stat Parameter] ) on row, ({Measure.[Stat Algorithm Parameter Association]}) on column include memberproperties {[Stat Parameter].[Stat Parameter], [Stat Parameter Weekly Default]} {[Stat Parameter].[Stat Parameter], [Stat Parameter Monthly Default]} {[Stat Parameter].[Stat Parameter], [Stat Parameter Quarterly Default]};

    BestFitSelectionCriteria : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row, ({Measure.[Bestfit Selection Criteria]}) on column;

    Weights : Select ([Version].[Version Name]* [Forecast Iteration].[Forecast Iteration] * [Item].[Segmentation LOB] ) on row,({Measure.[Straight Line Weight],Measure.[Trend Weight],Measure.[Seasonality Weight],Measure.[Level Weight],Measure.[Range Weight], Measure.[COCC Weight]}) on column;

    CurrentCycleBestfitAlgorithm : Select ([Version].[Version Name] * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] ) on row, ({Measure.[Bestfit Algorithm]}) on column;

    CustomL1WLagBacktest : Select ([Version].[Version Name] * [Time].[Week]  * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row,  ({Measure.[CML L1 W Lag Backtest]}) on column;

    CustomL1MLagBacktest : Select ([Version].[Version Name] * [Time].[Month]  * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row,  ({Measure.[CML L1 M Lag Backtest]}) on column;

    CustomL1PMLagBacktest : Select ([Version].[Version Name] * [Time].[Planning Month]  * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row,  ({Measure.[CML L1 PM Lag Backtest]}) on column;

    CustomL1QLagBacktest : Select ([Version].[Version Name] * [Time].[Quarter]  * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row,  ({Measure.[CML L1 Q Lag Backtest]}) on column;

    CustomL1PQLagBacktest : Select ([Version].[Version Name] * [Time].[Planning Quarter]  * [Region].[Stat Region] * [Item].[Stat Item] * [PnL].[Stat PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Stat Location] * [Channel].[Stat Channel] * [Planning Cycle].[Planning Cycle Date] * [Lag].[Lag] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] ) on row,  ({Measure.[CML L1 PQ Lag Backtest]}) on column;

    SystemEnsembleAlgorithmList : Select ([Forecast Iteration].[Forecast Iteration] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Version].[Version Name] * [Demand Domain].[Stat Demand Domain] * [Region].[Stat Region] * [Account].[Stat Account] * [Location].[Stat Location] * [Item].[Stat Item] ) on row, ({Measure.[System Ensemble Algorithm List]}) on column;

    EnsembleParameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row, ({Measure.[Ensemble Method], Measure.[Ensemble Top N Value]}) on column;

    IncludeEnsemble : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Stat Rule].[Stat Rule] ) on row, ({Measure.[Include Ensemble]}) on column;

    EnsembleWeights : Select ([Sequence].[Sequence] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Item].[Stat Item] * [Stat Algorithm].[Stat Algorithm] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [Location].[Stat Location] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] ) on row, ({Measure.[Planner Ensemble Weight], Measure.[Slice Association Stat]}) on column where {(Measure.[Slice Association Stat] == 1)};

    ForecastEngine : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row, ({Measure.[Forecast Engine]}) on column;

    SeasonalIndices :  Select ([Sequence].[Sequence] * [Forecast Iteration].[Forecast Iteration] * [Channel].[Stat Channel] * [Version].[Version Name] * [Account].[Stat Account] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Region].[Stat Region] * [Location].[Stat Location]  * [Item].[Stat Item] * [Time].[Partial Week]) on row, ({Measure.[SCHM Validation Seasonal Index Backtest],Measure.[SCHM Seasonal Index Backtest], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[SCHM Seasonal Index Backtest]), (Measure.[Slice Association Stat] == 1)};

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

    StatSegmentation : Select (Sequence.[Sequence] * Version.[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Planning Cycle].[Planning Cycle Date] *[Region].[Stat Region] * [Location].[Stat Location] * [Channel].[Stat Channel] *[PnL].[Stat PnL] * [Item].[Stat Item] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * Class.[Class] ) on row, ({Measure.[Volume Segment L1 Planning Cycle], Measure.[COV Segment L1 Planning Cycle],  Measure.[Product Segment L1 Planning Cycle],   Measure.[Length of Series L1 Planning Cycle], Measure.[Number of Zeros L1 Planning Cycle],  Measure.[Intermittent L1 Planning Cycle], Measure.[PLC Status L1 Planning Cycle], Measure.[Std Dev L1 Planning Cycle], Measure.[Avg Volume L1 Planning Cycle], Measure.[COV L1 Planning Cycle], Measure.[Volume L1 Planning Cycle], Measure.[Stat Disc Date System L1 Planning Cycle], Measure.[Stat Intro Date System L1 Planning Cycle],Measure.[Product Customer L1 Segment Planning Cycle], Measure.[Slice Association Stat]}) on column;

    ValidationParameters :   Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Validation Fold], Measure.[Validation Step Size],Measure.[Validation Period]}) on column;

    OutlierThresholds : Select (&CWV * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Outlier Absolute Threshold],Measure.[Outlier Percentage Threshold]}) on column;

    PlanningCycleDates : Select ([Planning Cycle].[Planning Cycle Date]) on row, () on column include memberproperties{[Planning Cycle].[Planning Cycle Date],Key};

    StatFcstCMLPlanningCycle : Select (Sequence.[Sequence] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Location].[Stat Location] * [Time].[Partial Week] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Item].[Stat Item] * [Demand Domain].[Stat Demand Domain] * [Planning Cycle].[Planning Cycle Date] * [Account].[Stat Account] ) on row,  ({Measure.[Stat Fcst CML Planning Cycle], Measure.[Slice Association Stat]}) on column;

Output Variables:
    Lag1FcstOutput
    LagNFcstOutput
    LagFcstOutput
    AllForecastWithLagDim
    SystemBestfitAlgorithmPlanningCycle
    PlanningCycleAlgoStats
    StabilityOutput
    ReasonabilityOutput

Slice Dimension Attributes:
    Sequence.[Sequence]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP027BacktestL1 import main

logger = logging.getLogger("o9_logger")

# Function Calls
StatLevelActual = O9DataLake.get("StatLevelActual")
Parameters = O9DataLake.get("Parameters")
SegmentationParameters = O9DataLake.get("SegmentationParameters")
OutlierParameters = O9DataLake.get("OutlierParameters")
AlgoParameters = O9DataLake.get("AlgoParameters")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
Rules = O9DataLake.get("Rules")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
MasterAlgoList = O9DataLake.get("MasterAlgoList")
DefaultAlgoParameters = O9DataLake.get("DefaultAlgoParameters")
BestFitSelectionCriteria = O9DataLake.get("BestFitSelectionCriteria")
CurrentCycleBestfitAlgorithm = O9DataLake.get("CurrentCycleBestfitAlgorithm")
CustomL1WLagBacktest = O9DataLake.get("CustomL1WLagBacktest")
CustomL1MLagBacktest = O9DataLake.get("CustomL1MLagBacktest")
CustomL1PMLagBacktest = O9DataLake.get("CustomL1PMLagBacktest")
CustomL1QLagBacktest = O9DataLake.get("CustomL1QLagBacktest")
CustomL1PQLagBacktest = O9DataLake.get("CustomL1PQLagBacktest")
SystemEnsembleAlgorithmList = O9DataLake.get("SystemEnsembleAlgorithmList")
EnsembleParameters = O9DataLake.get("EnsembleParameters")
IncludeEnsemble = O9DataLake.get("IncludeEnsemble")
EnsembleWeights = O9DataLake.get("EnsembleWeights")
ForecastEngine = O9DataLake.get("ForecastEngine")
SeasonalIndices = O9DataLake.get("SeasonalIndices")
SellOutOffset = O9DataLake.get("SellOutOffset")
StatSegmentation = O9DataLake.get("StatSegmentation")
ValidationParameters = O9DataLake.get("ValidationParameters")
StatFcstCMLPlanningCycle = O9DataLake.get("StatFcstCMLPlanningCycle")
OutlierThresholds = O9DataLake.get("OutlierThresholds")
PlanningCycleDates = O9DataLake.get("PlanningCycleDates")


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
    Lag1FcstOutput,
    LagNFcstOutput,
    LagFcstOutput,
    AllForecastWithLagDim,
    SystemBestfitAlgorithmPlanningCycle,
    PlanningCycleAlgoStats,
    StabilityOutput,
    ReasonabilityOutput,
) = main(
    StatLevelActual=StatLevelActual,
    Parameters=Parameters,
    SegmentationParameters=SegmentationParameters,
    OutlierParameters=OutlierParameters,
    AlgoParameters=AlgoParameters,
    TimeDimension=TimeDimension,
    CurrentTimePeriod=CurrentTimePeriod,
    Rules=Rules,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    StatBucketWeight=StatBucketWeight,
    StatSegmentation=StatSegmentation,
    ReadFromHive=ReadFromHive,
    RUN_SEGMENTATION_EVERY_CYCLE=RUN_SEGMENTATION_EVERY_CYCLE,
    RUN_VALIDATION_EVERY_FOLD=RUN_VALIDATION_EVERY_FOLD,
    RUN_BEST_FIT_EVERY_CYCLE=RUN_BEST_FIT_EVERY_CYCLE,
    Grains=Grains,
    UseHolidays=UseHolidays,
    IncludeDiscIntersections=IncludeDiscIntersections,
    OverrideFlatLineForecasts=OverrideFlatLineForecasts,
    MasterAlgoList=MasterAlgoList,
    df_keys=df_keys,
    alpha=AlphaACF,
    DefaultAlgoParameters=DefaultAlgoParameters,
    BestFitSelectionCriteria=BestFitSelectionCriteria,
    CurrentCycleBestfitAlgorithm=CurrentCycleBestfitAlgorithm,
    TrendVariationThreshold=TrendVariationThreshold,
    LevelVariationThreshold=LevelVariationThreshold,
    RangeVariationThreshold=RangeVariationThreshold,
    SeasonalVariationThreshold=SeasonalVariationThreshold,
    SeasonalVariationCountThreshold=SeasonalVariationCountThreshold,
    MinimumIndicePercentage=MinimumIndicePercentage,
    AbsTolerance=AbsTolerance,
    COCCVariationThreshold=COCCVariationThreshold,
    Weights=Weights,
    OutlierThresholds=OutlierThresholds,
    PlanningCycleDates=PlanningCycleDates,
    ReasonabilityCycles=ReasonabilityCycles,
    ACFLowerThreshold=ACFLowerThreshold,
    ACFUpperThreshold=ACFUpperThreshold,
    ACFSkipLags=ACFSkipLags,
    ACFDiff=ACFDiff,
    RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
    smooth_fraction=smooth_fraction,
    BackTestCyclePeriod=BackTestCyclePeriod,
    RolloverAlgorithmflag=RolloverAlgorithmflag,
    LagsToStore=LagsToStore,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    HistoryMeasure=HistoryMeasure,
    StatFcstCMLPlanningCycle=StatFcstCMLPlanningCycle,
    ValidationParameters=ValidationParameters,
    CustomL1WLagBacktest=CustomL1WLagBacktest,
    CustomL1MLagBacktest=CustomL1MLagBacktest,
    CustomL1PMLagBacktest=CustomL1PMLagBacktest,
    CustomL1QLagBacktest=CustomL1QLagBacktest,
    CustomL1PQLagBacktest=CustomL1PQLagBacktest,
    SystemEnsembleAlgorithmList=SystemEnsembleAlgorithmList,
    EnsembleParameters=EnsembleParameters,
    IncludeEnsemble=IncludeEnsemble,
    EnsembleWeights=EnsembleWeights,
    ForecastEngine=ForecastEngine,
    SeasonalIndices=SeasonalIndices,
    SellOutOffset=SellOutOffset,
    BacktestOnlyCustomAlgos=eval(BacktestOnlyCustomAlgos),
    BacktestOnlyStatAlgos=eval(BacktestOnlyStatAlgos),
    LagNInput=LagNInput,
    TrendThreshold=int(TrendThreshold),
)
O9DataLake.put("Lag1FcstOutput", Lag1FcstOutput)
O9DataLake.put("LagNFcstOutput", LagNFcstOutput)
O9DataLake.put("LagFcstOutput", LagFcstOutput)
O9DataLake.put("AllForecastWithLagDim", AllForecastWithLagDim)
O9DataLake.put("SystemBestfitAlgorithmPlanningCycle", SystemBestfitAlgorithmPlanningCycle)
O9DataLake.put("PlanningCycleAlgoStats", PlanningCycleAlgoStats)
O9DataLake.put("StabilityOutput", StabilityOutput)
O9DataLake.put("ReasonabilityOutput", ReasonabilityOutput)
