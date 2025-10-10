"""
Plugin : DP082NetChangeHistoryRealignment_And_Clustering
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Item],Location.[Location],Account.[Account],Region.[Region],Channel.[Channel],PnL.[PnL],Demand Domain.[Demand Domain]
    multiprocessing_num_cores - 4

Input Queries:
    AttributeMapping - Select ([Version].[Version Name] * [Data Object].[Data Object] ) on row,  ({Measure.[Data Object Account Level], Measure.[Data Object Channel Level], Measure.[Data Object Item Level], Measure.[Data Object Location Level], Measure.[Data Object Region Level],
        Measure.[Data Object PnL Level], Measure.[Data Object Demand Domain Level]}) on column;

    RealignmentRules - Select ([Version].[Version Name] * [Data Object].[Data Object] * [DM Rule].[Rule] * [Sequence].[Sequence] ) on row,
    ({Measure.[History Realignment Status], Measure.[Realignment Net Change Flag], Measure.[DP Realignment Rule Sequence],Measure.[Transition Start Date],
    Measure.[DP Realignment Cluster ID],Measure.[History Realignment Status Rerun],Measure.[DP Full History Realignment Status], Measure.[DP From Item Scope],
    Measure.[DP To Item Scope], Measure.[DP From Account Scope], Measure.[DP To Account Scope], Measure.[DP From Channel Scope], Measure.[DP To Channel Scope],
    Measure.[DP From Region Scope], Measure.[DP To Region Scope] , Measure.[DP From Location Scope],  Measure.[DP To Location Scope],  Measure.[DP From PnL Scope],
    Measure.[DP To PnL Scope], Measure.[DP From Demand Domain Scope], Measure.[DP To Demand Domain Scope], Measure.[DP From Item Scope LC],
    Measure.[DP To Item Scope LC], Measure.[DP From Account Scope LC], Measure.[DP To Account Scope LC], Measure.[DP From Channel Scope LC],
    Measure.[DP To Channel Scope LC], Measure.[DP From Region Scope LC], Measure.[DP To Region Scope LC] , Measure.[DP From Location Scope LC],
    Measure.[DP To Location Scope LC],  Measure.[DP From PnL Scope LC], Measure.[DP To PnL Scope LC], Measure.[DP From Demand Domain Scope LC],
    Measure.[DP To Demand Domain Scope LC]}) on column where { todatetime(Measure.[Transition Start date])<=todatetime(&CurrentDay.element(0).Key)};

    Actual - Select ([Version].[Version Name] * [Region].[Region] * [Item].[Item] * [Location].[Location] * [Demand Domain].[Demand Domain] * [Account].[Account] * [Channel].[Channel] * [PnL].[PnL] ) on row, ({Measure.[Actual Raw], Measure.[Actual Input]}) on column where {&AllPastDays};

    AccountMapping - select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]* [Account].[Account]) on row, () on column;

    ChannelMapping - select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] * [Channel].[Channel]) on row, () on column;

    PnLMapping - select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] * [PnL].[PnL]) on row, () on column;

    DemandDomainMapping - select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain]
    *[Demand Domain].[Demand Domain]) on row, () on column;

    LocationMapping - select ([Location].[Planning Location] * [Location].[Location]) on row, () on column;

    ItemMapping - select ([Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[All Item] * [Item].[Item]) on row, () on column;

    RegionMapping - select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region] * [Region].[Region]) on row, () on column;

Output Variables:
    CandidateOutput
    RealignmentOP
    RealignmentRuleAssociationOP

Slice Dimension Attributes:

"""

import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP082NetChangeHistoryRealignment_And_Clustering import main

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

# Function Calls
AttributeMapping = O9DataLake.get("AttributeMapping")
RealignmentRules = O9DataLake.get("RealignmentRules")
Actual = O9DataLake.get("Actual")
AccountMapping = O9DataLake.get("AccountMapping")
ChannelMapping = O9DataLake.get("ChannelMapping")
PnLMapping = O9DataLake.get("PnLMapping")
DemandDomainMapping = O9DataLake.get("DemandDomainMapping")
LocationMapping = O9DataLake.get("LocationMapping")
ItemMapping = O9DataLake.get("ItemMapping")
RegionMapping = O9DataLake.get("RegionMapping")


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

CandidateOutput, RealignmentOP, RealignmentRuleAssociationOP = main(
    Grains=Grains,
    AttributeMapping=AttributeMapping,
    RealignmentRules=RealignmentRules,
    Actual=Actual,
    AccountMapping=AccountMapping,
    ChannelMapping=ChannelMapping,
    PnLMapping=PnLMapping,
    DemandDomainMapping=DemandDomainMapping,
    LocationMapping=LocationMapping,
    ItemMapping=ItemMapping,
    RegionMapping=RegionMapping,
    df_keys=df_keys,
    multiprocessing_num_cores=int(multiprocessing_num_cores),
)
O9DataLake.put("CandidateOutput", CandidateOutput)
O9DataLake.put("RealignmentOP", RealignmentOP)
O9DataLake.put("RealignmentRuleAssociationOP", RealignmentRuleAssociationOP)
