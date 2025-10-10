"""
Plugin : DP057PopulateSplitPercent
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    HistoryMeasure - Actual
    Grains - Region.[Planning Region],Demand Domain.[Planning Demand Domain],Account.[Planning Account],Channel.[Planning Channel],PnL.[Planning PnL],Item.[Item],Location.[Location],Location.[Planning Location]
    TimeLevel - Time.[Partial Week]
    ReadFromHive - False
    MultiprocessingNumCores - 4

Input Queries:
    Actual : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Location] * [Location].[Planning Location] * [Item].[Item] * [Time].[Partial Week]) on row, ({Measure.[Actual], Measure.[Backorders], Measure.[Billing], Measure.[Orders], Measure.[Shipments]}) on column;

    CurrentTimePeriod : Select (&CurrentDay * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column;

    TimeDimension : Select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Partial Week], Key} {[Time].[Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

    LocationSplitParameters : Select ([Version].[Version Name]) on row, ({Measure.[Location Split History Period], Measure.[Location Split History Time Bucket]}) on column;

    LocationSplitPercentInputNormalized : Select ([Region].[Planning Region] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Time].[Partial Week] * [Item].[Item] * [Location].[Planning Location] * [Location].[Location] ) on row,  ({Measure.[Location Split % Input Normalized], Measure.[Location Split Assortment]}) on column where {Measure.[Location Split Assortment] == 1.0};

    ForecastBucket : Select (&DPSPForecastBuckets) on row, () on column include memberproperties {[Time].[Partial Week], Key};

    LocationSplitMethod : Select ([Region].[Planning Region] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Item].[Item] ) on row, ({Measure.[Location Split Method Final]}) on column;

    ItemConsensusFcst : Select ([Version].[Version Name] * [Location].[Planning Location] * [Item].[Item] * [Region].[Planning Region] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Time].[Partial Week] ) on row, ({Measure.[Item Consensus Fcst]}) on column;

    AssortmentFlag : Select ([Region].[Planning Region] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Item].[Item] * [Location].[Location] ) on row, ({Measure.[Assortment Final]}) on column where {Measure.[Assortment Final] > 0.0};

Output Variables:
    SKUSplit

PseudoCode:
    0) Merge LocationSplitPercentInputNormalized with TimeDimension to get partial week key
    1) Sort above dataframe on cust_grp, pl_item, location and partial_week_key to get from date and to dates
    2) Merge Actual with ItemDates to get intro and disc data
    3) Merge Actual and ItemConsensusFcst to get only relevant intersections
    4) Check if LocationSplitParameters is empty or not
    5) If not, we need to calculate "Location Split Moving Avg"
        5.1) Check for LocationSplitParameters["Location Split History Time Bucket"] and set relevant time name and key accordingly to get time_mapping
        5.2) Using "Location Split History Period" get MovingAvgPeriods and then get last_n_periods
        5.3) Using ForecastBucket get future_n_periods and FuturePeriods(len((future_n_periods)))
        5.4) Iterate through each customer group and item intersection in LocationSplitMethod
        5.5) From Actual, get data for customer group and item intersection
        5.6) Merge it with TimeDimension and filter data according to intro_date and disc_date and get relevant data
        5.7) Now we have relevant actual and using groupby get aggregated data at relevant time level
        5.8) Calculate moving average for future_n_periods
        5.9) Aggregate actual at item level and use this to get moving average split value
        5.10) Merge it with TimeDimension to get data at partial week level (copy to leaves)
        5.11) Filter the data to get partial weeks present in ForecastBucket
        5.12) Merge the resulted dataframe with AssortmentFlag and then normalize the moving average split values
    6) Else, we need to calculate "Location Split Fixed %" only
        6.1) Merge LocationSplitPercentInputNormalized with ItemAttribute
        6.2) Create cartesian product of LocationSplitPercentInputNormalized with partial week and partial week key
        6.3) Filter for only those partial weeks which lies in between from_date and to_date and also in between start_date and end_date
        6.4) Calculate cumulative sum of split %, using group by and transform
        6.5) Normalize the split % by dividing split % with cumulative sum
    7) Merge both the output dataframes and get the resultant output


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
ItemDates = O9DataLake.get("ItemDates")

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

from helpers.DP057PopulateSplitPercent import main

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
    multiprocessing_num_cores=int(MultiprocessingNumCores),
)
logger.info("Writing output to o9DataLake ...")
O9DataLake.put("SKUSplit", SKUSplit)
