"""
Plugin : DP088ValidateRealignmentRulesFromFileUpload
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Input Queries:
    AttributeMapping - Select ([Version].[Version Name] * [Data Object].[Data Object] ) on row,  ({Measure.[Data Object Account Level], Measure.[Data Object Channel Level], Measure.[Data Object Item Level], Measure.[Data Object Location Level], Measure.[Data Object Region Level], Measure.[Data Object PnL Level], Measure.[Data Object Demand Domain Level]}) on column;

    FileUpload - Select ([Version].[Version Name] * [Personnel].[Email] * [Sequence].[Sequence] ) on row,
    ({Measure.[DP Realignment Data Object Input], Measure.[DP Rule Description Input], Measure.[DP From Account Scope UI Input], Measure.[DP To Account Scope UI Input], Measure.[DP From Channel Scope UI Input], Measure.[DP To Channel Scope UI Input], Measure.[DP From Demand Domain Scope UI Input], Measure.[DP To Demand Domain Scope UI Input], Measure.[DP From Item Scope UI Input], Measure.[DP To Item Scope UI Input], Measure.[DP From Location Scope UI Input], Measure.[DP To Location Scope UI Input], Measure.[DP From PnL Scope UI Input], Measure.[DP To PnL Scope UI Input], Measure.[DP From Region Scope UI Input], Measure.[DP To Region Scope UI Input], Measure.[DP Rule Sequence Input], Measure.[DP Realignment Percentage Input], Measure.[DP Conversion Factor Input], Measure.[History Realignment Active Period Input], Measure.[Transition Start Date Input], Measure.[Transition End Date Input]}) on column;

    ItemMapping - select ([Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4] * [Item].[L5] * [Item].[L6] * [Item].[All Item]) on row, () on column include memberproperties {Item.[Planning Item], DisplayName} {Item.[L1], DisplayName} {Item.[L2], DisplayName} {Item.[L3], DisplayName} {Item.[L4], DisplayName} {Item.[L5], DisplayName} {Item.[L6], DisplayName} {Item.[All Item], DisplayName};

    RegionMapping - select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region]) on row, () on column include memberproperties {[Region].[Planning Region], DisplayName} {[Region].[Region L1], DisplayName} {[Region].[Region L2], DisplayName}  {[Region].[Region L3], DisplayName} {[Region].[Region L4], DisplayName} {[Region].[All Region], DisplayName};

    AccountMapping - select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column include memberproperties {[Account].[Account L1], DisplayName} {[Account].[Account L2], DisplayName} {[Account].[Account L3], DisplayName} {[Account].[Account L4], DisplayName} {[Account].[All Account], DisplayName} {[Account].[Planning Account], DisplayName};

    ChannelMapping - select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] * [Channel].[Channel]) on row, () on column include memberproperties {[Channel].[Channel L1], DisplayName} {[Channel].[Channel L2], DisplayName} {[Channel].[Planning Channel], DisplayName} {[Channel].[All Channel], DisplayName} {[Channel].[Channel], DisplayName};

    PnLMapping - select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column include memberproperties {[PnL].[All PnL], DisplayName} {[PnL].[Planning PnL], DisplayName} {[PnL].[PnL L1], DisplayName} {[PnL].[PnL L2], DisplayName} {[PnL].[PnL L3], DisplayName} {[PnL].[PnL L4], DisplayName};

    DemandDomainMapping - select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Demand Domain]) on row, () on column include memberproperties {[Demand Domain].[All Demand Domain], DisplayName} {[Demand Domain].[Demand Domain L1], DisplayName} {[Demand Domain].[Demand Domain L2], DisplayName} {[Demand Domain].[Demand Domain L3], DisplayName} {[Demand Domain].[Demand Domain L4], DisplayName} {[Demand Domain].[Planning Demand Domain], DisplayName} {[Demand Domain].[Demand Domain], DisplayName};

    LocationMapping - select ([Location].[Location] * [Location].[Planning Location]) on row, () on column include memberproperties {Location.[Location], DisplayName} {Location.[Planning Location], DisplayName};

    CurrentDay - Select (&CurrentDay) on row, () on column include memberproperties {[Time].[Day], Key};

Output Variables:
    RemarksOP, ValidRulesOP
"""

import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP088ValidateRealignmentRulesFromFileUpload import main
from helpers.o9helpers.o9logger import O9Logger

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = O9Logger()

# Function Calls
AttributeMapping = O9DataLake.get("AttributeMapping")
FileUpload = O9DataLake.get("FileUpload")
AccountMapping = O9DataLake.get("AccountMapping")
ChannelMapping = O9DataLake.get("ChannelMapping")
PnLMapping = O9DataLake.get("PnLMapping")
DemandDomainMapping = O9DataLake.get("DemandDomainMapping")
LocationMapping = O9DataLake.get("LocationMapping")
ItemMapping = O9DataLake.get("ItemMapping")
RegionMapping = O9DataLake.get("RegionMapping")
CurrentDay = O9DataLake.get("CurrentDay")

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

RemarksOP, ValidRulesOP = main(
    AttributeMapping=AttributeMapping,
    FileUpload=FileUpload,
    AccountMapping=AccountMapping,
    ChannelMapping=ChannelMapping,
    PnLMapping=PnLMapping,
    DemandDomainMapping=DemandDomainMapping,
    LocationMapping=LocationMapping,
    ItemMapping=ItemMapping,
    RegionMapping=RegionMapping,
    CurrentDay=CurrentDay,
    df_keys=df_keys,
)

O9DataLake.put("RemarksOP", RemarksOP)
O9DataLake.put("ValidRulesOP", ValidRulesOP)
