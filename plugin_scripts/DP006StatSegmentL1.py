"""
Plugin : DP006StatSegmentL1
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:
    DimClass - Class.[Class]
    VolSegLvl - A,B
    COVSegLvl - X,Y
    VolumeThresholdMeasure - Volume Threshold
    COVThresholdMeasure - COV Threshold
    ReadFromHive - False
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    SegmentationVolumeCalcGrain - Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location],Item.[Segmentation LOB]
    PlannerOverrideCycles - None
    BacktestCycle - None
    RUN_SEGMENTATION_EVERY_FOLD - False
    RUN_SEGMENTATION_EVERY_CYCLE - False
    RUN_SEGMENTATION_FORWARD_CYCLE - True

Input Queries:
    SegmentThresholds : Select ([Class].[Class] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Volume Threshold], Measure.[COV Threshold]}) on column;

    MiscThresholds : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row, ({Measure.[Intermittency Threshold], Measure.[New Launch Period], Measure.[Disco Period], Measure.[History Time Buckets], Measure.[History Measure],Measure.[Volume-COV History Period], Measure.[History Period]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    Actual : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Item].[Stat Item] * [Item].[Segmentation LOB] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Time].[Partial Week]   * {Measure.[Stat Actual]}) ;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

    PlanningCycleDates : Select([Planning Cycle].[Planning Cycle Date]) on row, () on column include memberproperties{[Planning Cycle].[Planning Cycle Date],Key};

    ValidationParameters :  Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Validation Fold], Measure.[Validation Step Size],Measure.[Validation Period]}) on column;

Output Variables:
    StatSegmentation
    ClassAssociationOutput
    ItemClass

Slice Dimension Attributes: None

Pseudocode :

    * segmentation_level - grains at which certain metrics like (Volume L1, Cumulative Vol L1) is to be calculated

    * segmentation_output_level - grain at which output is expected.

    * combine segmentation_level and segmentation_output_level to be included in the groupby and form 'common_grains'

    * join LastTimePeriod with PW mapping to get the Partial Week Key corresponding to last partial week

    * use the Last Partial Week Key to filter out Actuals prior to this date

    * split on delimiter and extract the thresholds/levels for volume and COV

    * extract the forecast generation time bucket - this could be Week/Month/Planning Month/Quarter/Planning Quarter

    * join the actuals (at partial week level) with time dimension based on forecast generation time bucket

    * aggregate actuals up to the forecast generation time bucket level

    * fill the missing dates in the series and impute 0 in such intersections

    * group the intersections into MATURE/NPI/EOL based on the steps below

    * use the NPI and EOL threshold periods (say 12 months) to calculate the cut off dates for both

    * if an intersection has not sold in last 12 months, that will be termed as EOL

    * if an intersection has not sold before last 12 months, that will be termed as NPI and rest of the intersections will be mature

    * for volume and cov calculation, we need to filter actuals based on the parameter "Volume-COV History Period"

    * filter data for the relevant periods and calculate all the required measures (Volume L1, Volume % L1, Cumulative Volume L1, COV etc) at an intersection level

    * based on volume and cov, intersections are classified into let's say AX, BX, AY, BY (among the mature items)

    * calculate attributes like length of series, intermittency

    * collect the required output measures and create the output dataframe

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP006StatSegmentL1 import main

logger = logging.getLogger("o9_logger")

# Function Calls

Actual = O9DataLake.get("Actual")
SegmentThresholds = O9DataLake.get("SegmentThresholds")
MiscThresholds = O9DataLake.get("MiscThresholds")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
SellOutOffset = O9DataLake.get("SellOutOffset")
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

StatSegmentation, ClassAssociationOutput, ItemClass = main(
    COVSegLvl=COVSegLvl,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    COVThresholdMeasure=COVThresholdMeasure,
    Actual=Actual,
    Grains=Grains,
    BacktestCycle=BacktestCycle,
    PlannerOverrideCycles=PlannerOverrideCycles,
    RUN_SEGMENTATION_EVERY_FOLD=RUN_SEGMENTATION_EVERY_FOLD,
    RUN_SEGMENTATION_EVERY_CYCLE=RUN_SEGMENTATION_EVERY_CYCLE,
    RUN_SEGMENTATION_FORWARD_CYCLE=RUN_SEGMENTATION_FORWARD_CYCLE,
    VolumeThresholdMeasure=VolumeThresholdMeasure,
    DimClass=DimClass,
    SegmentThresholds=SegmentThresholds,
    ValidationParameters=ValidationParameters,
    MiscThresholds=MiscThresholds,
    TimeDimension=TimeDimension,
    VolSegLvl=VolSegLvl,
    CurrentTimePeriod=CurrentTimePeriod,
    SellOutOffset=SellOutOffset,
    PlanningCycleDates=PlanningCycleDates,
    ReadFromHive=ReadFromHive,
    SegmentationVolumeCalcGrain=SegmentationVolumeCalcGrain,
    df_keys=df_keys,
)

O9DataLake.put("StatSegmentation", StatSegmentation)
O9DataLake.put("ClassAssociationOutput", ClassAssociationOutput)
O9DataLake.put("ItemClass", ItemClass)
