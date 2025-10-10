"""
Plugin : DP028PlanningLevelProfile
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    ItemLevel - Item.[Planning Item]
    SalesDomainLevel - Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain]
    LocationLevel - Location.[Planning Location]
    ReadFromHive - False
    InputTables - Actual,SellOutActual
    OutputMeasure - Stat Fcst Final Profile PL
    HistoryPeriodsInWeeks - 13,52,104,156
    MultiprocessingNumCores - 4
    DefaultMapping - Item-All Planning Item,Account-All Planning Account,Location-All Planning Location,Region-All Planning Region,Channel-All Planning Channel,Demand Domain-All Planning Demand Domain,PnL-All Planning PnL
    NBucketsinMonths - 12

Input Queries:
    Parameters : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[History Measure], Measure.[History Period], Measure.[Forecast Period]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    Actual : Select ([Sequence].[Sequence] * [Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Time].[Partial Week] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Transition Demand Domain] * [Demand Domain].[Planning Demand Domain]  * [Location].[Planning Location] * [Item].[Planning Item] * [Item].[Transition Item] ) on row, ({Measure.[Actual], Measure.[Slice Association PL]}) on column where {~isnull(Measure.[Actual]), (Measure.[Slice Association PL] == 1)};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    PItemMetaData : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Transition Demand Domain] * [Demand Domain].[Planning Demand Domain] * [Item].[Planning Item] * [Item].[Transition Item]) on row, ({Measure.[Intro Date], Measure.[Disco Date], Measure.[Phase Out Profile], Measure.[Number of Phase Out Buckets], Measure.[Adjust Phase Out Profile], Measure.[Phase In Split %], Measure.[Product Transition Overlap Start Date], Measure.[Transition Type], Measure.[Assortment Phase In]}) on column;

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]  ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Time].[Partial Week] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    TransitionFlag : Select ([Version].[Version Name] * FROM.[PnL].[Planning PnL] * FROM.[Item].[Planning Item] * FROM.[Demand Domain].[Planning Demand Domain] * FROM.[Account].[Planning Account] * FROM.[Region].[Planning Region] * FROM.[Channel].[Planning Channel] * FROM.[Location].[Planning Location] * TO.[Item].[Planning Item]) on row, ({Edge.[510 Product Transition].[Transition Flag]}) on column where {RelationshipType.[510 Product Transition]};

    ForecastIterationMasterData : Select (Version.[Version Name] * [Forecast Iteration].[Forecast Iteration Type] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Iteration Type Input Stream]}) on column;

    SellOutActual :  Select ([Sequence].[Sequence] *[Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] *  [Item].[Transition Item] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain] * [Account].[Planning Account] ) on row,  ({Measure.[Sell Out Actual], Measure.[Slice Association PL]}) on column where {(Measure.[Slice Association PL] == 1)};

    SliceAssociation : Select ([Version].[Version Name] * [Account].[Planning Account] * [Item].[Transition Item] * [Location].[Planning Location] * [Region].[Planning Region] * [Channel].[Planning Channel] * [Sequence].[Sequence] * [PnL].[Planning PnL] * [Demand Domain].[Transition Demand Domain] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Slice Association TL]}) on column;

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

    DefaultProfiles : Select ([Version].[Version Name] * [Lifecycle Time].[Lifecycle Bucket] * [PLC Profile].[PLC Profile] ) on row, ({Measure.[Default Profile]}) on column;


Output Variables:
    PLProfile

Slice Dimension Attributes:
    Sequence.[Sequence]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.dataframe_utils import convert_category_to_str
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP028PlanningLevelProfile import main

logger = logging.getLogger("o9_logger")


# Function Calls
Parameters = O9DataLake.get("Parameters")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
PItemMetaData = O9DataLake.get("PItemMetaData")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
TransitionFlag = O9DataLake.get("TransitionFlag")
ForecastIterationMasterData = O9DataLake.get("ForecastIterationMasterData")
SellOutOffset = O9DataLake.get("SellOutOffset")
DefaultProfiles = O9DataLake.get("DefaultProfiles")


DefaultMapping_list = DefaultMapping.strip().split(",")
default_mapping = {}
for mapping in DefaultMapping_list:
    key, value = mapping.split("-")
    default_mapping[key] = value

# configurable inputs
try:
    CombinedActual = {}
    AllInputTables = InputTables.strip().split(",")
    for i in AllInputTables:
        Input = O9DataLake.get(i)
        if Input is None:
            logger.exception(f"Input '{i}' is None, please recheck the table name...")
            continue
        Input = convert_category_to_str(Input)
        CombinedActual[i] = Input
        logger.warning(f"Reading input {i}...")
except Exception as e:
    logger.exception(f"Error while reading inputs\nError:-\n{e}")

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

PLProfile, ActualLastNBucketsPL = main(
    LocationLevel=LocationLevel,
    ItemLevel=ItemLevel,
    Actual=CombinedActual,
    TimeDimension=TimeDimension,
    PItemDates=PItemMetaData,
    Parameters=Parameters,
    SalesDomainLevel=SalesDomainLevel,
    CurrentTimePeriod=CurrentTimePeriod,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    StatBucketWeight=StatBucketWeight,
    ReadFromHive=ReadFromHive,
    OutputMeasure=OutputMeasure,
    HistoryPeriodsInWeeks=HistoryPeriodsInWeeks,
    MultiprocessingNumCores=int(MultiprocessingNumCores),
    TransitionFlag=TransitionFlag,
    ForecastIterationMasterData=ForecastIterationMasterData,
    SellOutOffset=SellOutOffset,
    NBucketsinMonths=NBucketsinMonths,
    default_mapping=default_mapping,
    DefaultProfiles=DefaultProfiles,
    df_keys=df_keys,
)
O9DataLake.put("PLProfile", PLProfile)
O9DataLake.put("ActualLastNBucketsPL", ActualLastNBucketsPL)
