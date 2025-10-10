"""
Plugin : DP084InnovationFrameworkItemSplit
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    HistoryMeasure - Actual
    Grains - Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location],Item.[Item]
    TimeLevel - Time.[Partial Week]
    MultiprocessingNumCores - 4

Input Queries:
    Actual : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Item].[Item] * [Time].[Partial Week]) on row, ({Measure.[Actual]}) on column;

    CurrentTimePeriod : Select (&CurrentDay * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column;

    TimeDimension : Select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Partial Week], Key} {[Time].[Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

    ItemMapping : Select ([Item].[Item] * [Item].[Planning Item]) on row, () on column include memberproperties {[Item].[Item], "Item Intro Date"} {[Item].[Item], "Item Disc Date"};

    ItemSplitParameters : Select ([Version].[Version Name]) on row, ({Measure.[Item Split History Period], Measure.[Item Split History Time Bucket]}) on column;

    ForecastBucket : Select (&DPSPForecastBuckets) on row, () on column include memberproperties {[Time].[Partial Week], Key};

    ItemSplitMethod : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Item].[Planning Item] ) on row, ({Measure.[Item Split Method Final]}) on column;

    NPIFcstL1 : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Partial Week] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [PnL].[Planning PnL] * [Initiative].[Initiative] * [Region].[Planning Region] ) on row, ({Measure.[Planning Level NPI Fcst L1]}) on column;

    AssortmentFlag : Select ([Version].[Version Name] * [Initiative].[Initiative] * [Item].[Item] * [Channel].[Planning Channel] * [Account].[Planning Account] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] ) on row, ({Measure.[NPI Assortment Final]}) on column where {Measure.[NPI Assortment Final] > 0.0};

Output Variables:
    SKUSplit

PseudoCode:

"""

import logging
from threading import Thread

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP084InnovationFrameworkItemSplit import main

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
ItemMapping = O9DataLake.get("ItemMapping")
ForecastBucket = O9DataLake.get("ForecastBucket")
ItemSplitMethod = O9DataLake.get("ItemSplitMethod")
NPIFcstL1 = O9DataLake.get("NPIFcstL1")
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

SKUSplit = main(
    HistoryMeasure=HistoryMeasure,
    Grains=Grains,
    TimeLevel=TimeLevel,
    Actual=Actual,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    df_keys=df_keys,
    ItemSplitParameters=ItemSplitParameters,
    ItemMapping=ItemMapping,
    ForecastBucket=ForecastBucket,
    ItemSplitMethod=ItemSplitMethod,
    NPIFcstL1=NPIFcstL1,
    AssortmentFlag=AssortmentFlag,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
)

logger.info("Writing output to o9DataLake ...")
O9DataLake.put("SKUSplit", SKUSplit)
