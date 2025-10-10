"""
Plugin : DP052PopulateSKUSplitPercent
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    HistoryMeasure - Actual
    Grains - Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location],Item.[Item]
    TimeLevel - Time.[Partial Week]
    ReadFromHive - False
    MultiprocessingNumCores - 4

Input Queries:
    Actual : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Item].[Item] * [Time].[Partial Week]) on row, ({Measure.[Actual], Measure.[Backorders], Measure.[Billing], Measure.[Orders], Measure.[Shipments]}) on column;

    CurrentTimePeriod : Select (&CurrentDay * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column;

    TimeDimension : Select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Partial Week], Key} {[Time].[Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

    ItemMapping : Select ([Item].[Item] * [Item].[Planning Item]) on row, () on column include memberproperties {[Item].[Item], "Item Intro Date"} {[Item].[Item], "Item Disc Date"};

    ItemSplitParameters : Select ([Version].[Version Name]) on row, ({Measure.[Item Split History Period], Measure.[Item Split History Time Bucket]}) on column;

    ItemSplitPercentInputNormalized : Select ([Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Version].[Version Name] * [Time].[Partial Week] * [Item].[Item] ) on row,  ({Measure.[Item Split % Input Normalized], Measure.[Item Split Assortment]}) on column where {Measure.[Item Split Assortment] == 1.0};

    ForecastBucket : Select (&DPSPForecastBuckets) on row, () on column include memberproperties {[Time].[Partial Week], Key};

    ItemSplitMethod : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Item].[Planning Item] ) on row, ({Measure.[Item Split Method Final]}) on column;

    ConsensusFcst : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Time].[Partial Week] * [Item].[Planning Item] ) on row, ({Measure.[Consensus Fcst]}) on column;

    AssortmentFlag : Select ([Region].[Planning Region] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Item].[Item] ) on row, ({Measure.[Assortment Final]}) on column where {Measure.[Assortment Final] > 0.0};

Output Variables:
    SKUSplit

PseudoCode:
    0) Merge ItemSplitPercentInputNormalized with TimeDimension to get partial week key
    1) Sort the above dataframe on cust_grp, item and partial_week_key to get from date and to dates
    2) Fill null values of intro with start date of TimeDimension and disc with end date of TimeDimension
    3) Merge actual with ItemMapping to get planning item and then merge it with AssortmentFlag to get only those intersections where assortment flag is 1
    4) Get unique customer group and planning items combination from actual - cust_grp_pl_item_in_actual
    5) Cartesian product ItemMapping with CustomerGroupDimension to get all possible combinations - cust_grp_pl_item
    6) All combinations in cust_grp_pl_item which are not in cust_grp_pl_item_in_actual are new - new_intersections
        6.1) Get item count for each combination of customer group and planning item
        6.2) Calculate split% for new intersection (1/item_count)
    7) Check if ItemSplitParameters is empty or not
    8) If not, we need to calculate "Item Split Moving Avg"
        8.1) Check for ItemSplitParameters["Item Split History Time Bucket"] and set relevant time accordingly
        8.2) Using "Item Split History Period" get MovingAvgPeriods and then get the last_n_periods
        8.3) Using ForecastBucket get future_n_periods and FuturePeriods(len((future_n_periods)))
        8.4) For every customer group and planning item combination in ItemSplitParameters, calculate "Item Split Moving Avg"
        8.5) Get actual_df using customer group and planning item combination
        8.6) Filter actual_df where "Actuals" is present for day between intro_date and disc_date and also for last_n_periods only to get relevant_actual
        8.7) Aggregate the "Actuals" from day level to relevant_time level
        8.8) Calculate moving average for future_n_periods
        8.9) Map output_data with ItemMapping to get planning item and aggregate actual from item level to planning item level
        8.10) Use actual data and aggregated actual data to calculate split %, for items whose actual data is present
        8.11) Merge it with TimeDimension to get data at partial week level (copy to leaves)
        8.12) Filter the data to get partial weeks present in ForecastBucket
        8.13) Merge the resulted dataframe with ConsensusFcst and then normalize the moving average split values
    9) If len(new_intersections) != 0
        9.1) Cartesian product this with partial week and partial week key col
        9.2) Filter out only partial weeks present in DPSPForecastBuckets and satisfy intro and disc dates
    10) Concat the new_intersections and output_data (take care of empty dataframe case)
    11) Else, we need to calculate "Item Split Fixed %" only
        11.1) Merge ItemSplitPercentInputNormalized with ItemAttribute
        11.2) Create cartesian product of ItemSplitPercentInputNormalized with partial week and partial week key
        11.3) Filter for only those partial weeks which lies in between from_date and to_date and also in between start_date and end_date
        11.4) Calculate cumulative sum of split %, using group by and transform
        11.5) Normalize the split % by dividing split % with cumulative sum
    12) Merge both the output dataframes and get the resultant output


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
ItemSplitParameters = O9DataLake.get("ItemSplitParameters")
ItemSplitPercentInputNormalized = O9DataLake.get("ItemSplitPercentInputNormalized")
ItemMapping = O9DataLake.get("ItemMapping")
ForecastBucket = O9DataLake.get("ForecastBucket")
ItemSplitMethod = O9DataLake.get("ItemSplitMethod")
ConsensusFcst = O9DataLake.get("ConsensusFcst")
AssortmentFlag = O9DataLake.get("AssortmentFlag")

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

from helpers.DP052PopulateSKUSplitPercent import main

SKUSplit = main(
    HistoryMeasure=HistoryMeasure,
    Grains=Grains,
    TimeLevel=TimeLevel,
    Actual=Actual,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    df_keys=df_keys,
    ItemSplitParameters=ItemSplitParameters,
    ItemSplitPercentInputNormalized=ItemSplitPercentInputNormalized,
    ItemMapping=ItemMapping,
    ForecastBucket=ForecastBucket,
    ItemSplitMethod=ItemSplitMethod,
    ConsensusFcst=ConsensusFcst,
    AssortmentFlag=AssortmentFlag,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
)
logger.info("Writing output to o9DataLake ...")
O9DataLake.put("SKUSplit", SKUSplit)
