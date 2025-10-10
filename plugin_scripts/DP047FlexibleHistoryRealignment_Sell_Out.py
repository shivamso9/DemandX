"""
Plugin : DP047FlexibleHistoryRealignment_Sell_Out
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Input Queries:
// Full Actual
Select ([Version].[Version Name] * [Time].[Week] * [Time].[Day] * [Region].[Region] * [Item].[Item] * [Demand Domain].[Demand Domain] * [Account].[Account] * [Channel].[Channel]) on row,
({Measure.[Sell Out Actual Raw],Measure.[Ch INV Actual Raw]}) on column;

// Actual
Select ([Version].[Version Name] * [Time].[Week] * &HistoryRefreshBuckets * [Region].[Region] * [Item].[Item] * [Demand Domain].[Demand Domain] * [Account].[Account] * [Channel].[Channel] ) on row,
({Measure.[Sell Out Actual Raw],Measure.[Ch INV Actual Raw]}) on column where {(coalesce(Measure.[Sell Out Actual Raw],0) < 0)};

//AttributeMapping
Select ([Version].[Version Name] * [Data Object].[Data Object] ) on row,  ({Measure.[Data Object Account Level], Measure.[Data Object Channel Level], Measure.[Data Object Item Level],
Measure.[Data Object Location Level], Measure.[Data Object Region Level], Measure.[Data Object PnL Level], Measure.[Data Object Demand Domain Level]}) on column;

//RealignmentRules
Select ([Version].[Version Name] * [Data Object].[Data Object] * [DM Rule].[Rule] * [Sequence].[Sequence] ) on row,  ({Measure.[DP From Item Scope], Measure.[DP To Item Scope],
Measure.[DP From Account Scope], Measure.[DP To Account Scope], Measure.[DP From Channel Scope], Measure.[DP To Channel Scope], Measure.[DP From Region Scope], Measure.[DP To Region Scope] ,
Measure.[DP From Location Scope],Measure.[DP To Location Scope],Measure.[DP From PnL Scope], Measure.[DP To PnL Scope], Measure.[DP From Demand Domain Scope], Measure.[DP To Demand Domain Scope],
Measure.[DP Realignment Percentage],Measure.[DP Conversion Factor],Measure.[DP Full History Realignment Status],Measure.[History Realignment Status System],Measure.[DP Realignment Rule Sequence],
Measure.[Transition Start Date], Measure.[History Realignment Active Period]}) on column;

//HistoryRefreshBuckets
Select (&HistoryRefreshBuckets) on row, () on column;

//ItemMaster
select ([Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[All Item] * [Item].[Item]) on row, () on column;

//RegionMaster
select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region] * [Region].[Region]) on row, () on column;

//AccountMaster
select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]* [Account].[Account]) on row, () on column;

//ChannelMaster
select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] * [Channel].[Channel]) on row, () on column;

//PnLMaster
select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] * [PnL].[PnL]) on row, () on column;

//DemandDomainMaster
select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain] *[Demand Domain].[Demand Domain]) on row, () on column;

//LocationMaster
select ([Location].[Location] * [Location].[Planning Location]) on row, () on column;

//TimeDimension
Select (&AllPastDays.relatedmembers([Week]) * [Time].[Day]) on row, () on column include memberproperties {Time.[Day], Key};

// KeyFigures
Select ([Version].[Version Name] * [Data Object].[Data Object Type].[Sell Out History Measures] * [Data Object].[Data Object] ) on row, ({Measure.[Include in History Realignment]}) on column;

Output Variables:
RealignmentOP, ActualOP

Slice Dimensions: "Time.[Week]"
"""

import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP047FlexibleHistoryRealignment_Sell_In_Sell_Out import main

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

# Function Calls
Actual_df = O9DataLake.get("Actual")
Full_Actual_df = O9DataLake.get("FullActual")
AttributeMapping_df = O9DataLake.get("AttributeMapping")
HistoryRefreshBuckets_df = O9DataLake.get("HistoryRefreshBuckets")
RealignmentRules_df = O9DataLake.get("RealignmentRules")
ItemMaster_df = O9DataLake.get("ItemMaster")
RegionMaster_df = O9DataLake.get("RegionMaster")
AccountMaster_df = O9DataLake.get("AccountMaster")
ChannelMaster_df = O9DataLake.get("ChannelMaster")
PnlMaster_df = O9DataLake.get("PnlMaster")
DemandDomainMaster_df = O9DataLake.get("DemandDomainMaster")
LocationMaster_df = O9DataLake.get("LocationMaster")
TimeDimension_df = O9DataLake.get("TimeDimension")
KeyFigures_df = O9DataLake.get("KeyFigures")

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

RealignmentOP, ActualOP = main(
    Actual=Actual_df,
    Full_Actual=Full_Actual_df,
    AttributeMapping=AttributeMapping_df,
    HistoryRefreshBuckets=HistoryRefreshBuckets_df,
    RealignmentRules=RealignmentRules_df,
    ItemMaster=ItemMaster_df,
    RegionMaster=RegionMaster_df,
    AccountMaster=AccountMaster_df,
    ChannelMaster=ChannelMaster_df,
    PnlMaster=PnlMaster_df,
    DemandDomainMaster=DemandDomainMaster_df,
    LocationMaster=LocationMaster_df,
    TimeDimension=TimeDimension_df,
    KeyFigures=KeyFigures_df,
    df_keys=df_keys,
)

O9DataLake.put("RealignmentOP", RealignmentOP)
O9DataLake.put("ActualOP", ActualOP)
