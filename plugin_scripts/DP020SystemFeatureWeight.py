"""
Plugin : DP020SystemFeatureWeight
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:

    FeatureLevel : Item.[L4]

    FeatureNames : Item.[L1],Item.[L2],Item.[L3],Item.[L4],Item.[L5],Item.[Stat Item],Item.[L6],Item.[Item Stage],Item.[Item Type],Item.[Item Class],Item.[Transition Group],Item.[A1],Item.[A2],Item.[A3],Item.[A4],Item.[A5],Item.[A6],Item.[A7],Item.[A8],Item.[A9],Item.[A10],Item.[A11],Item.[A12],Item.[A13],Item.[A14],Item.[A15],Item.[Planning Item Locale],Item.[Base UOM],Item.[UOM Type],Item.[Container Size],Item.[Package Size],Item.[All Item]

    ReadFromHive : False

    Frequency : Weekly

    NumericalCols : None

Input Queries:
    ItemAttribute : Select ([Item].[Item]*[Item].[Planning Item]*[Item].[Transition Item]*[Item].[Stat Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]*[Item].[Item Stage]*[Item].[Item Type]*[Item].[PLC Status]*[Item].[Item Class]*[Item].[Transition Group]*[Item].[A1]*[Item].[A2]*[Item].[A3]*[Item].[A4]*[Item].[A5]*[Item].[A6]*[Item].[A7]*[Item].[A8]*[Item].[A9]*[Item].[A10]*[Item].[A11]*[Item].[A12]*[Item].[A13]*[Item].[A14]*[Item].[A15]*[Item].[All Item]) on row,() on column include memberproperties {[Item].[Item],[Item Intro Date]}{[Item].[Item],[Item Disc Date]}{[Item].[Item],[Item Status]}{[Item].[Item],[Is New Item]} INCLUDE_NULLMEMBERS;

    PlanningItemAttribute : Select ([Item].[Planning Item]) on row, () on column include memberproperties {[Item].[Planning Item],[Planning Item Intro Date]}{[Item].[Planning Item],[Planning Item Disc Date]}{[Item].[Planning Item],[Planning Item Locale]}{[Item].[Planning Item],[Base UOM]}{[Item].[Planning Item],[UOM Type]}{[Item].[Planning Item],[Container Size]}{[Item].[Planning Item],[Package Size]} INCLUDE_NULLMEMBERS;

    Sales : Select ([Item].[Item] * [Time].[Day] * [Version].[Version Name]) on row, ({Measure.[Like Item Actual]}) on column;

    SearchSpace : Select ([Version].[Version Name] ) on row,  ({Measure.[Like Item Search Space]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

Output Variables:
    FeatureWeight

Slice Dimension Attributes:


"""

import logging

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

from o9_common_utils.O9DataLake import O9DataLake

logger = logging.getLogger("o9_logger")

import threading

from o9Reference.common_utils.o9_memory_utils import _get_memory

# Function Calls
ItemAttribute = O9DataLake.get("ItemAttribute")
Sales = O9DataLake.get("Sales")
PlanningItemAttribute = O9DataLake.get("PlanningItemAttribute")
SearchSpace = O9DataLake.get("SearchSpace")
TimeDimension = O9DataLake.get("TimeDimension")

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

from helpers.DP020SystemFeatureWeight import main

FeatureWeight = main(
    Item=ItemAttribute,
    sales=Sales,
    Frequency=Frequency,
    pl_item=PlanningItemAttribute,
    SearchSpace=SearchSpace,
    TimeDimension=TimeDimension,
    NumericalCols=NumericalCols,
    FeatureLevel=FeatureLevel,
    FeatureNames=FeatureNames,
    ReadFromHive=ReadFromHive,
    df_keys=df_keys,
)
O9DataLake.put("FeatureWeight", FeatureWeight)
