"""
Plugin : DP085InnovationFrameworkLocationSplit
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    HistoryMeasure - Actual
    Grains - Region.[Planning Region],Demand Domain.[Planning Demand Domain],Account.[Planning Account],Channel.[Planning Channel],PnL.[Planning PnL],Item.[Item],Location.[Planning Location],Location.[Location]
    TimeLevel - Time.[Partial Week]
    MultiprocessingNumCores - 4

Input Queries:
    Actual : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Location] *  [Location].[Planning Location] * [Item].[Item] * [Time].[Partial Week]) on row, ({Measure.[Actual]}) on column;

    CurrentTimePeriod : Select (&CurrentDay * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column;

    TimeDimension : Select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Partial Week], Key} {[Time].[Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

    LocationSplitParameters : Select ([Version].[Version Name]) on row, ({Measure.[Location Split History Period], Measure.[Location Split History Time Bucket]}) on column;

    ForecastBucket : Select (&DPSPForecastBuckets) on row, () on column include memberproperties {[Time].[Partial Week], Key};

    LocationSplitMethod : Select ([Region].[Planning Region] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Item].[Item] ) on row, ({Measure.[Location Split Method Final]}) on column;

    ItemNPIFcst : Select ([Version].[Version Name] * [Item].[Item] * [Channel].[Planning Channel] * [Time].[Partial Week] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [PnL].[Planning PnL] * [Initiative].[Initiative] * [Region].[Planning Region] ) on row, ({Measure.[Initiative Item NPI Fcst]}) on column;

    AssortmentFlag : Select ([Version].[Version Name] * [Initiative].[Initiative] * [Item].[Item] * [Channel].[Planning Channel] * [Account].[Planning Account] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Location].[Location] ) on row, ({Measure.[NPI Assortment Final]}) on column where {Measure.[NPI Assortment Final] > 0.0};

    ItemAttribute : Select ([Item].[Item]*[Item].[Planning Item]*[Item].[Transition Item]*[Item].[Stat Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]*[Item].[Item Stage]*[Item].[Item Type]*[Item].[PLC Status]*[Item].[Item Class]*[Item].[Transition Group]*[Item].[A1]*[Item].[A2]*[Item].[A3]*[Item].[A4]*[Item].[A5]*[Item].[A6]*[Item].[A7]*[Item].[A8]*[Item].[A9]*[Item].[A10]*[Item].[A11]*[Item].[A12]*[Item].[A13]*[Item].[A14]*[Item].[A15]*[Item].[All Item]) on row,() on column include memberproperties {[Item].[Item],[Item Intro Date]}{[Item].[Item],[Item Disc Date]}{[Item].[Item],[Item Status]}{[Item].[Item],[Is New Item]} INCLUDE_NULLMEMBERS;

Output Variables:
    SKUSplit

PseudoCode:

"""

import logging
from threading import Thread

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP085InnovationFrameworkLocationSplit import main

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
ForecastBucket = O9DataLake.get("ForecastBucket")
LocationSplitMethod = O9DataLake.get("LocationSplitMethod")
ItemNPIFcst = O9DataLake.get("ItemNPIFcst")
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

SKUSplit = main(
    HistoryMeasure=HistoryMeasure,
    Grains=Grains,
    TimeLevel=TimeLevel,
    Actual=Actual,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    df_keys=df_keys,
    LocationSplitParameters=LocationSplitParameters,
    ForecastBucket=ForecastBucket,
    LocationSplitMethod=LocationSplitMethod,
    ItemNPIFcst=ItemNPIFcst,
    AssortmentFlag=AssortmentFlag,
    ItemAttribute=ItemAttribute,
    multiprocessing_num_cores=int(1),
)

logger.info("Writing output to o9DataLake ...")
O9DataLake.put("SKUSplit", SKUSplit)
