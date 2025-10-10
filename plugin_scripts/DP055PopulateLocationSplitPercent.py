"""
Plugin : DP055PopulateLocationSplitPercent
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    HistoryMeasure - Actual
    Grains - Region.[Planning Region],Demand Domain.[Planning Demand Domain],Account.[Planning Account],Channel.[Planning Channel],PnL.[Planning PnL],Item.[Item],Location.[Planning Location],Location.[Location]
    TimeLevel - Time.[Partial Week]
    ReadFromHive - False
    MultiprocessingNumCores - 4

Input Queries:
    Actual : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Location] *  [Location].[Planning Location] * [Item].[Item] * [Time].[Partial Week]) on row, ({Measure.[Actual], Measure.[Backorders], Measure.[Billing], Measure.[Orders], Measure.[Shipments]}) on column;

    CurrentTimePeriod : Select (&CurrentDay * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column;

    TimeDimension : Select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Partial Week], Key} {[Time].[Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

    LocationSplitParameters : Select ([Version].[Version Name]) on row, ({Measure.[Location Split History Period], Measure.[Location Split History Time Bucket]}) on column;

    LocationSplitPercentInputNormalized : Select ([Region].[Planning Region] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Time].[Partial Week] * [Item].[Item] * [Location].[Planning Location] * [Location].[Location] ) on row,  ({Measure.[Location Split % Input Normalized], Measure.[Location Split Assortment]}) on column where {Measure.[Location Split Assortment] == 1.0};

    ForecastBucket : Select (&DPSPForecastBuckets) on row, () on column include memberproperties {[Time].[Partial Week], Key};

    LocationSplitMethod : Select ([Region].[Planning Region] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Item].[Item] ) on row, ({Measure.[Location Split Method Final]}) on column;

    ItemConsensusFcst : Select ([Version].[Version Name] * [Location].[Planning Location] * [Item].[Item] * [Region].[Planning Region] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Time].[Partial Week] ) on row, ({Measure.[Item Consensus Fcst]}) on column;

    AssortmentFlag : Select ([Region].[Planning Region] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Item].[Item] * [Location].[Location] ) on row, ({Measure.[Assortment Final]}) on column where {Measure.[Assortment Final] > 0.0};

    ItemAttribute - Select ([Item].[Item]*[Item].[Planning Item]*[Item].[Transition Item]*[Item].[Stat Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]*[Item].[Item Stage]*[Item].[Item Type]*[Item].[PLC Status]*[Item].[Item Class]*[Item].[Transition Group]*[Item].[A1]*[Item].[A2]*[Item].[A3]*[Item].[A4]*[Item].[A5]*[Item].[A6]*[Item].[A7]*[Item].[A8]*[Item].[A9]*[Item].[A10]*[Item].[A11]*[Item].[A12]*[Item].[A13]*[Item].[A14]*[Item].[A15]*[Item].[All Item]) on row,() on column include memberproperties {[Item].[Item],[Item Intro Date]}{[Item].[Item],[Item Disc Date]}{[Item].[Item],[Item Status]}{[Item].[Item],[Is New Item]} INCLUDE_NULLMEMBERS;

Output Variables:
    SKUSplit

PseudoCode:
    0) Merge LocationSplitPercentInputNormalized with TimeDimension to get partial week key
    1) Sort above dataframe on dimensions and partial_week_key to get from date and to dates
    2) Define search level
    3) Get relevant columns in the item dataframe -> ItemAttribute
    4) If dates are null, fill intro with start date of TimeDimension and disc with end date of TimeDimension
    5) Merge Actual and ItemAttribute
    6) Get unique customer group, item combinations present in ConsensusFcst dataframe - intersections_with_consensus_fcst
    7) Get unique customer group, item combinations present in Actual dataframe - intersections_with_actual
    8) Merge intersections_with_consensus_fcst and intersections_with_actual with indicator column - merged_df
    9) Merge the merged_df with LocationSplitMethod to get intersections for which we want to generate output
    10) Check if LocationSplitParameters is empty or not
    11) If not, we need to calculate "Location Split Moving Avg"
        11.1) Check for LocationSplitParameters["Location Split History Time Bucket"] and set relevant time name and key accordingly to get time_mapping
        11.2) Using "Location Split History Period" get MovingAvgPeriods and then get last_n_periods
        11.3) Using ForecastBucket get future_n_periods and FuturePeriods(len((future_n_periods)))
        11.4) Create two different dataframes - 1) indicator = "left_only" - intersections_with_consensus_fcst_df
                                                2) indicator = "both" - intersections_with_consensus_fcst_and_actual_df
        11.5) Check if intersections_with_consensus_fcst_df is empty or not
        11.6) For intersections_with_consensus_fcst_df, means it has no actual and we need to go up in hierarchy to get tha actual data
            11.6.1) Merge Actual with intersections_with_consensus_fcst_df
            11.6.2) Now iterate through each level in search_level to get actual data at that search_level
            11.6.3) Use Assortment_flag dataframe to get locations to process for particular item
            11.6.4) Cartesian product it with time_mapping to get data at relevant time level
        11.7) For intersections_with_consensus_fcst_and_actual_df, means it has actual
            11.7.1) Merge Actual with intersections_with_consensus_fcst_and_actual_df
            11.7.2) Merge it with TimeDimension to get relevant time name
        11.8) Concat 11.6 and 11.7 outputs to get relevant_intersections and using groupby get aggregated data at relevant time level
        11.9) Calculate moving average for future_n_periods
        11.10) Aggregate actual at item level and use this to get moving average split value
        11.11) Merge it with TimeDimension to get data at partial week level (copy to leaves)
        11.12) Filter the data to get partial weeks present in ForecastBucket
        11.13) Merge the resulted dataframe with AssortmentFlag and then normalize the moving average split values
    12) Else, we need to calculate "Location Split Fixed %" only
            12.1) Merge LocationSplitPercentInputNormalized with ItemAttribute
            12.2) Create cartesian product of LocationSplitPercentInputNormalized with partial week and partial week key
            12.3) Filter for only those partial weeks which lies in between from_date and to_date and also in between start_date and end_date
            12.4) Calculate cumulative sum of split %, using group by and transform
            12.5) Normalize the split % by dividing split % with cumulative sum
    13) Merge both the output dataframes and get the resultant output


"""

import logging
from threading import Thread

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


# Function Calls
# Read Inputs
logger.info("Reading data from o9DataLake ...")
Actual = O9DataLake.get("Actual")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
LocationSplitParameters = O9DataLake.get("LocationSplitParameters")
LocationSplitPercentInputNormalized = O9DataLake.get("LocationSplitPercentInputNormalized")
ForecastBucket = O9DataLake.get("ForecastBucket")
LocationSplitMethod = O9DataLake.get("LocationSplitMethod")
ItemConsensusFcst = O9DataLake.get("ItemConsensusFcst")
AssortmentFlag = O9DataLake.get("AssortmentFlag")
ItemAttribute = O9DataLake.get("ItemAttribute")

# Check if slicing variable is present
if "df_keys" not in locals():
    logging.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}
else:
    logger.info("Slice : {}".format(df_keys))

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.
back_thread = Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

from helpers.DP055PopulateLocationSplitPercent import main

SKUSplit = main(
    HistoryMeasure=HistoryMeasure,
    Grains=Grains,
    TimeLevel=TimeLevel,
    Actual=Actual,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    df_keys=df_keys,
    LocationSplitParameters=LocationSplitParameters,
    LocationSplitPercentInputNormalized=LocationSplitPercentInputNormalized,
    ForecastBucket=ForecastBucket,
    LocationSplitMethod=LocationSplitMethod,
    ItemConsensusFcst=ItemConsensusFcst,
    AssortmentFlag=AssortmentFlag,
    ItemAttribute=ItemAttribute,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
)
logger.info("Writing output to o9DataLake ...")
O9DataLake.put("SKUSplit", SKUSplit)
