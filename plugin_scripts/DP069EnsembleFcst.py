"""
    Plugin : DP069EnsembleFcst
    Version : 2025.08.00
    Maintained by : dpref@o9solutions.com

    Script Params:
        EnsembleOnlyStatAlgos - False
        EnsembleOnlyCustomAlgos - False
        Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
        HistoryMeasure - Stat Actual
        OverrideFlatLineForecasts - False
        TrendVariationThreshold - 0.6
        LevelVariationThreshold - 0.6
        RangeVariationThreshold - 0.6
        SeasonalVariationThreshold - 0.6
        SeasonalVariationCountThreshold - 0.6
        ReasonabilityCycles - 1
        MinimumIndicePercentage - 0.05
        AbsTolerance - 50
        COCCVariationThreshold - 0.1
        MultiprocessingNumCores - 4

    Input Queries:
        TimeDimension : Select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

        CurrentTimePeriod : Select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

        ForecastParameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[History Period], Measure.[Forecast Period], Measure.[Validation Period], Measure.[Bestfit Method], Measure.[Error Metric], Measure.[History Time Buckets], Measure.[Forecast Strategy]}) on column;

        ActualsAndForecastData : Select ([Sequence].[Sequence] *  [Forecast Iteration].[Forecast Iteration] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Actual], Measure.[Actual Cleansed], Measure.[Stat Fcst AR-NNET], Measure.[Stat Fcst Auto ARIMA], Measure.[Stat Fcst Croston], Measure.[Stat Fcst DES], Measure.[Stat Fcst ETS], Measure.[Stat Fcst Moving Average], Measure.[Stat Fcst Naive Random Walk], Measure.[Stat Fcst Prophet], Measure.[Stat Fcst SES], Measure.[Stat Fcst STLF], Measure.[Stat Fcst Seasonal Naive YoY], Measure.[Stat Fcst TBATS], Measure.[Stat Fcst TES], Measure.[Stat Fcst Theta], Measure.[Stat Fcst sARIMA], Measure.[Stat Fcst Simple Snaive], Measure.[Stat Fcst Weighted Snaive], Measure.[Stat Fcst Growth Snaive], Measure.[Stat Fcst Simple AOA], Measure.[Stat Fcst Weighted AOA], Measure.[Stat Fcst Growth AOA],Measure.[Stat Fcst CML], Measure.[Slice Association Stat]}) on column where {(Measure.[Slice Association Stat] == 1)};

        ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

        AssignedAlgoList :  Select ([Sequence].[Sequence] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Item].[Stat Item] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] * [PnL].[Stat PnL] ) on row, ({Measure.[Assigned Algorithm List], Measure.[Planner Bestfit Algorithm], Measure.[System Assigned Algorithm List], Measure.[Assigned Rule], Measure.[Planner Assigned Algorithm List], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Assigned Algorithm List]), (Measure.[Slice Association Stat] == 1)};

        Weights : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Item].[Segmentation LOB] ) on row,({Measure.[Straight Line Weight],Measure.[Trend Weight],Measure.[Seasonality Weight],Measure.[Level Weight],Measure.[Range Weight], Measure.[COCC Weight]}) on column;

        SelectionCriteria : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,({Measure.[Bestfit Selection Criteria]}) on column;

        Violations : Select ([Sequence].[Sequence] * [Forecast Iteration].[Forecast Iteration] * [Stat Rule].[Stat Rule] * [Stat Algorithm].[Stat Algorithm] * [Version].[Version Name] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Demand Domain].[Stat Demand Domain] * [PnL].[Stat PnL] * [Region].[Stat Region] * [Location].[Stat Location]* [Item].[Segmentation LOB] * [Item].[Stat Item] ) on row,({Measure.[Straight Line],Measure.[Trend Violation],Measure.[Seasonal Violation],Measure.[Level Violation],Measure.[Range Violation],Measure.[COCC Violation], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Straight Line]), (Measure.[Slice Association Stat] == 1)};

        EnsembleParameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row, ({Measure.[Ensemble Method], Measure.[Ensemble Top N Value]}) on column;

        IncludeEnsemble : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Stat Rule].[Stat Rule] ) on row, ({Measure.[Include Ensemble]}) on column;

        AlgoStats : Select ([Sequence].[Sequence] *  [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Location].[Stat Location] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Item].[Stat Item] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Stat Rule].[Stat Rule] * [Stat Algorithm].[Stat Algorithm] ) on row, ({Measure.[Validation Error], Measure.[Composite Error], Measure.[Validation Method], Measure.[Slice Association Stat]}) on column where {(Measure.[Slice Association Stat] == 1)};

        EnsembleWeights : Select ([Sequence].[Sequence] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Item].[Stat Item] * [Stat Algorithm].[Stat Algorithm] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [Location].[Stat Location] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] ) on row, ({Measure.[Planner Ensemble Weight], Measure.[Slice Association Stat]}) on column where {(Measure.[Slice Association Stat] == 1)};

        StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

        ForecastData : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Time].[Partial Week] * [Item].[Stat Item] ) on row, ({Measure.[Planner Stat Fcst L1], Measure.[Stat Fcst AR-NNET], Measure.[Stat Fcst Auto ARIMA], Measure.[Stat Fcst Croston], Measure.[Stat Fcst DES], Measure.[Stat Fcst ETS], Measure.[Stat Fcst Growth AOA], Measure.[Stat Fcst Growth Snaive], Measure.[Stat Fcst Moving Average], Measure.[Stat Fcst Naive Random Walk], Measure.[Stat Fcst Prophet], Measure.[Stat Fcst SES], Measure.[Stat Fcst STLF], Measure.[Stat Fcst Seasonal Naive YoY], Measure.[Stat Fcst Simple AOA], Measure.[Stat Fcst Simple Snaive], Measure.[Stat Fcst TBATS], Measure.[Stat Fcst TES], Measure.[Stat Fcst Theta], Measure.[Stat Fcst Weighted AOA], Measure.[Stat Fcst Weighted Snaive], Measure.[Stat Fcst sARIMA], Measure.[Stat Fcst ML L1], Measure.[Stat Fcst L1], Measure.[Slice Association Stat], Measure.[Stat Bucket Weight], Measure.[Stat Fcst L1 LC]}) on column where {~isnull(Measure.[Stat Bucket Weight]), (Measure.[Slice Association Stat] == 1)};

        SegmentationOutput : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Item].[Stat Item] ) on row, ({Measure.[Intermittent L1], Measure.[PLC Status L1], Measure.[Seasonality L1], Measure.[Length of Series L1], Measure.[Bestfit Algorithm], Measure.[Planner Bestfit Algorithm], Measure.[Assigned Rule], Measure.[Planner Assigned Algorithm List], Measure.[System Assigned Algorithm List], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[PLC Status L1]), (Measure.[Slice Association Stat] == 1)};

        MasterAlgoList : Select ([Version].[Version Name]) on row, ({Measure.[Assigned Algorithm List]}) on column;

        SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

        ForecastEngine : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Engine]}) on column;

        PlanningCycleDates : Select ([Planning Cycle].[Planning Cycle Date]) on row, ({Measure.[PlanningCycleDateKey]}) on column;

        ValidationParameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Validation Fold], Measure.[Validation Step Size], Measure.[Validation Period]}) on column;

    Output Variables:
        BestFitAlgo, ValidationError, FcstNextNBuckets, AllEnsembleDesc, AllEnsembleFcst, SystemEnsembleAlgorithmList, OutputAllAlgo

    Slice Dimension Attributes:
        Sequence.[Sequence]
    """

import logging

from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

import threading

from o9_common_utils.O9DataLake import O9DataLake

# Function Calls
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
ForecastParameters = O9DataLake.get("ForecastParameters")
ActualsAndForecastData = O9DataLake.get("ActualsAndForecastData")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
AssignedAlgoList = O9DataLake.get("AssignedAlgoList")
Weights = O9DataLake.get("Weights")
SelectionCriteria = O9DataLake.get("SelectionCriteria")
Violations = O9DataLake.get("Violations")
EnsembleParameters = O9DataLake.get("EnsembleParameters")
IncludeEnsemble = O9DataLake.get("IncludeEnsemble")
AlgoStats = O9DataLake.get("AlgoStats")
EnsembleWeights = O9DataLake.get("EnsembleWeights")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
ForecastData = O9DataLake.get("ForecastData")
SegmentationOutput = O9DataLake.get("SegmentationOutput")
MasterAlgoList = O9DataLake.get("MasterAlgoList")
SellOutOffset = O9DataLake.get("SellOutOffset")
ForecastEngine = O9DataLake.get("ForecastEngine")
PlanningCycleDates = O9DataLake.get("PlanningCycleDates")
ValidationParameters = O9DataLake.get("ValidationParameters")

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

from helpers.DP069EnsembleFcst import main

(
    BestFitAlgo,
    ValidationError,
    FcstNextNBuckets,
    AllEnsembleDesc,
    AllEnsembleFcst,
    SystemEnsembleAlgorithmList,
    OutputAllAlgo,
) = main(
    TimeDimension=TimeDimension,
    CurrentTimePeriod=CurrentTimePeriod,
    ForecastParameters=ForecastParameters,
    ActualsAndForecastData=ActualsAndForecastData,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    AssignedAlgoList=AssignedAlgoList,
    Weights=Weights,
    SelectionCriteria=SelectionCriteria,
    Violations=Violations,
    EnsembleParameters=EnsembleParameters,
    IncludeEnsemble=IncludeEnsemble,
    EnsembleWeights=EnsembleWeights,
    AlgoStats=AlgoStats,
    MasterAlgoList=MasterAlgoList,
    ForecastData=ForecastData,
    SegmentationOutput=SegmentationOutput,
    StatBucketWeight=StatBucketWeight,
    OverrideFlatLineForecasts=OverrideFlatLineForecasts,
    EnsembleOnlyStatAlgos=eval(EnsembleOnlyStatAlgos),
    EnsembleOnlyCustomAlgos=eval(EnsembleOnlyCustomAlgos),
    Grains=Grains,
    HistoryMeasure=HistoryMeasure,
    SellOutOffset=SellOutOffset,
    ForecastEngine=ForecastEngine,
    PlanningCycleDates=PlanningCycleDates,
    ValidationParameters=ValidationParameters,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    TrendVariationThreshold=float(TrendVariationThreshold),
    LevelVariationThreshold=float(LevelVariationThreshold),
    RangeVariationThreshold=float(RangeVariationThreshold),
    SeasonalVariationThreshold=float(SeasonalVariationThreshold),
    SeasonalVariationCountThreshold=float(SeasonalVariationCountThreshold),
    ReasonabilityCycles=float(ReasonabilityCycles),
    MinimumIndicePercentage=float(MinimumIndicePercentage),
    AbsTolerance=float(AbsTolerance),
    COCCVariationThreshold=float(COCCVariationThreshold),
    df_keys=df_keys,
)

O9DataLake.put("BestFitAlgo", BestFitAlgo)
O9DataLake.put("ValidationError", ValidationError)
O9DataLake.put("FcstNextNBuckets", FcstNextNBuckets)
O9DataLake.put("AllEnsembleDesc", AllEnsembleDesc)
O9DataLake.put("AllEnsembleFcst", AllEnsembleFcst)
O9DataLake.put("SystemEnsembleAlgorithmList", SystemEnsembleAlgorithmList)
O9DataLake.put("OutputAllAlgo", OutputAllAlgo)
