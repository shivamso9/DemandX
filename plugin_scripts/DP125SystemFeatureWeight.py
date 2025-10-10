"""
Plugin : DP125SystemFeatureWeight
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    FeatureLevel : Item.[L1]

    FeatureNames : Item.[L1],Item.[L2],Item.[L3],Item.[L4],Item.[L5],Item.[Stat Item],Item.[L6],Item.[Item Stage],Item.[Item Type],Item.[Item Class],Item.[Transition Group],Item.[A1],Item.[A2],Item.[A3],Item.[A4],Item.[A5],Item.[A6],Item.[A7],Item.[A8],Item.[A9],Item.[A10],Item.[A11],Item.[A12],Item.[A13],Item.[A14],Item.[A15],Item.[Planning Item Locale],Item.[Base UOM],Item.[UOM Type],Item.[Container Size],Item.[Package Size],Item.[All Item]

    ReadFromHive : False

    Frequency : Weekly

    NumericalCols : None

    ModelConfig: Controls model performance. Options: 'Stable', 'Balanced'(default), 'Fast', 'VeryFast'.
        A faster model may have lower accuracy.

    CloudStorageType: Cloud Storage Type. Default 'google'.

    MultiprocessingNumCores: None

Input Queries:
    ItemAttribute : Select ([Item].[Item]*[Item].[Planning Item]*[Item].[Transition Item]*[Item].[Stat Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]*[Item].[Item Stage]*[Item].[Item Type]*[Item].[PLC Status]*[Item].[Item Class]*[Item].[Transition Group]*[Item].[A1]*[Item].[A2]*[Item].[A3]*[Item].[A4]*[Item].[A5]*[Item].[A6]*[Item].[A7]*[Item].[A8]*[Item].[A9]*[Item].[A10]*[Item].[A11]*[Item].[A12]*[Item].[A13]*[Item].[A14]*[Item].[A15]*[Item].[All Item]) on row,() on column include memberproperties {[Item].[Item],[Item Intro Date]}{[Item].[Item],[Item Disc Date]}{[Item].[Item],[Item Status]}{[Item].[Item],[Is New Item]} INCLUDE_NULLMEMBERS;

    PlanningItemAttribute : Select ([Item].[Planning Item]) on row, () on column include memberproperties {[Item].[Planning Item],[Planning Item Intro Date]}{[Item].[Planning Item],[Planning Item Disc Date]}{[Item].[Planning Item],[Planning Item Locale]}{[Item].[Planning Item],[Base UOM]}{[Item].[Planning Item],[UOM Type]}{[Item].[Planning Item],[Container Size]}{[Item].[Planning Item],[Package Size]} INCLUDE_NULLMEMBERS;

    Sales : Select ([Version].[Version Name] * [Time].[Day] * [Region].[Region] * [Item].[Item] * [PnL].[PnL] * [Location].[Location] * [Channel].[Channel] * [Account].[Account] * [Demand Domain].[Demand Domain] ) on row,  ({Measure.[Like Item Actual]}) on column;

    SearchSpace : Select ([Version].[Version Name] * [Data Object].[Data Object] * [DM Rule].[Rule] ) on row,
    ({Measure.[Like Account Search Space by Level], Measure.[Like Channel Search Space by Level], Measure.[Like Demand Domain Search Space by Level], Measure.[Like Item Search Space by Level], Measure.[Like Location Search Space by Level], Measure.[Like PnL Search Space by Level], Measure.[Like Region Search Space by Level]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

    AccountAttribute: Select ([Account].[All Account] * [Account].[Account L4] * [Account].[Account L3] * [Account].[Account L2] * [Account].[Account L1] * [Account].[Planning Account] * [Account].[Account] * [Account].[Stat Account] * [Account].[Stat Account Group]);

    LocationAttribute: Select ([Location].[All Location]*[Location].[Stat Location Group]*[Location].[Stat Location]*[Location].[Location Type]*[Location].[Location Country]*[Location].[Location Region]*[Location].[Planning Location]*[Location].[Reporting Location]*[Location].[Location]);

    ChannelAttribute: Select ( [Channel].[All Channel] * [Channel].[Stat Channel Group] * [Channel].[Stat Channel] * [Channel].[Channel L2] * [Channel].[Channel L1] * [Channel].[Planning Channel] * [Channel].[Channel]);

    RegionAttribute: Select ([Region].[All Region]*[Region].[Stat Region]*[Region].[Stat Region Group]*[Region].[Region L4]*[Region].[Region L3]*[Region].[Region L2]*[Region].[Region L1]*[Region].[Planning Region]*[Region].[Region]);

    PnLAttribute: Select ([PnL].[All PnL]*[PnL].[Stat PnL Group]*[PnL].[Stat PnL]*[PnL].[PnL L4]*[PnL].[PnL L3]*[PnL].[PnL L2]*[PnL].[PnL L1]*[PnL].[Planning PnL]*[PnL].[PnL]);

    DemandDomainAttribute: Select ([Demand Domain].[All Demand Domain]*[Demand Domain].[Stat Demand Domain Group]*[Demand Domain].[Stat Demand Domain]*[Demand Domain].[Transition Demand Domain]*[Demand Domain].[Demand Domain L4]*[Demand Domain].[Demand Domain L3]*[Demand Domain].[Demand Domain L2]*[Demand Domain].[Demand Domain L1]*[Demand Domain].[Planning Demand Domain]*[Demand Domain].[Demand Domain]);

Output Variables:
    System Feature Weight by Level

Slice Dimension Attributes:

"""

import logging
import os

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
PlanningItemAttribute = O9DataLake.get("PlanningItemAttribute")
Sales = O9DataLake.get("Sales")
SearchSpace = O9DataLake.get("SearchSpace")
TimeDimension = O9DataLake.get("TimeDimension")
AccountAttribute = O9DataLake.get("AccountAttribute")
LocationAttribute = O9DataLake.get("LocationAttribute")
ChannelAttribute = O9DataLake.get("ChannelAttribute")
RegionAttribute = O9DataLake.get("RegionAttribute")
PnLAttribute = O9DataLake.get("PnLAttribute")
DemandDomainAttribute = O9DataLake.get("DemandDomainAttribute")

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

from helpers.DP125SystemFeatureWeight import main, save_output_data_to_cloud

FeatureWeight, FeatureWeightsFull = main(
    Item=ItemAttribute,
    Location=LocationAttribute,
    Account=AccountAttribute,
    Channel=ChannelAttribute,
    DemandDomain=DemandDomainAttribute,
    PnL=PnLAttribute,
    Region=RegionAttribute,
    sales=Sales,
    Frequency=Frequency,
    pl_item=PlanningItemAttribute,
    SearchSpace=SearchSpace,
    TimeDimension=TimeDimension,
    NumericalCols=NumericalCols,
    FeatureLevel=FeatureLevel,
    FeatureNames=FeatureNames,
    ReadFromHive=ReadFromHive,
    Model_Config=ModelConfig,
    MultiprocessingNumCores=MultiprocessingNumCores,
    df_keys=df_keys,
)


save_output_data_to_cloud(FeatureWeightsFull, df_keys, CloudStorageType)
O9DataLake.put("FeatureWeight", FeatureWeight)
