"""
Plugin : DP042StoreGroupAssortments_testing
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Input Queries:
    LocationGroupMaintenance : Select ( [Version].[Version Name] * [Data Object].[Data Object Type].[Assortment Management] * [DM Rule].[Rule] ) on row,  ({Measure.[DP Item Level], Measure.[DP Item Scope], Measure.[DP Location Level], Measure.[DP Location Scope], Measure.[DP Location Group Scope], Measure.[DP Region Level], Measure.[DP Region Scope], Measure.[DP Channel Level], Measure.[DP Channel Scope], Measure.[DP Account Level], Measure.[DP Account Scope], Measure.[DP PnL Level], Measure.[DP PnL Scope], Measure.[DP Demand Domain Level], Measure.[DP Demand Domain Scope], Measure.[DP Rule Sequence], Measure.[DP Rule Created By], Measure.[DP Rule Created Date]}) on column where {[Data Object].[Data Object].[Assortment Management L1]};

    ItemLocationGroupAssortment : Select ( [Version].[Version Name] * [Data Object].[Data Object Type] * [DM Rule].[Rule] ) on row,  ({Measure.[DP Item Level], Measure.[DP Item Scope], Measure.[DP Location Level], Measure.[DP Location Scope], Measure.[DP Location Group Scope], Measure.[DP Region Level], Measure.[DP Region Scope], Measure.[DP Channel Level], Measure.[DP Channel Scope], Measure.[DP Account Level], Measure.[DP Account Scope], Measure.[DP PnL Level], Measure.[DP PnL Scope], Measure.[DP Demand Domain Level], Measure.[DP Demand Domain Scope], Measure.[DP Exclude Flag], Measure.[DP Intro Date], Measure.[DP Disco Date], Measure.[DP Rule Sequence], Measure.[DP Rule Created By], Measure.[DP Rule Created Date]}) on column;

    DimItem : Select ([Item].[Item]*[Item].[Planning Item]*[Item].[Transition Item]*[Item].[Stat Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]);

    StoreDates : select ([Location].[Location] * [Item].[L3] * [Cluster].[Cluster]) on row, () on column include memberproperties {[Location].[Location],[Store Intro Date]} {[Location].[Location],[Store Disc Date]};

    DimLocation : select ([Location].[Location Region] * [Location].[Location Country] * [Location].[Reporting Location] * [Location].[Planning Location] * [Location].[Location]);

    DimChannel : select([Channel].[All Channel].[All] * [Channel].[Channel] * [Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] * [Channel].[Stat Channel]);

    DimPnL : select([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] * [PnL].[Stat PnL]);

    DimDemandDomain : select([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain] * [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] * [Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] * [Demand Domain].[Stat Demand Domain]);

    DimRegion : select([Region].[All Region] * [Region].[Planning Region] * [Region].[Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[Stat Region]);

    DimAccount : select([Account].[All Account] * [Account].[Account] * [Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[Planning Account] * [Account].[Stat Account]);

Output Variables:
    ActiveAssortmentByLocationGroup
    AssortmentByDatesGroup

Slice Dimension Attributes: None
"""

import logging
import threading

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP042StoreGroupAssortments import main
from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")

# Function Calls
Input1 = O9DataLake.get("LocationGroupMaintenance")
Input2 = O9DataLake.get("ItemLocationGroupAssortment")
Input3 = O9DataLake.get("I3")
DimItem = O9DataLake.get("DimItem")
StoreDates = O9DataLake.get("StoreDates")
DimLocation = O9DataLake.get("DimLocation")
DimAccount = O9DataLake.get("DimAccount")
DimChannel = O9DataLake.get("DimChannel")
DimDemandDomain = O9DataLake.get("DimDemandDomain")
DimPnL = O9DataLake.get("DimPnL")
DimRegion = O9DataLake.get("DimRegion")

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

ActiveAssortmentByLocationGroup, AssortmentByDatesGroup = main(
    Input1=Input1,
    Input2=Input2,
    Input3=Input3,
    DimItem=DimItem,
    StoreDates=StoreDates,
    DimLocation=DimLocation,
    DimAccount=DimAccount,
    DimRegion=DimRegion,
    DimChannel=DimChannel,
    DimPnL=DimPnL,
    DimDemandDomain=DimDemandDomain,
)


O9DataLake.put("ActiveAssortmentByLocationGroup", ActiveAssortmentByLocationGroup)
O9DataLake.put("AssortmentByDatesGroup", AssortmentByDatesGroup)
