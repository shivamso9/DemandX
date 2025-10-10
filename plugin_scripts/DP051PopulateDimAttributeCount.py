"""
Plugin : DP051PopulateDimAttributeCount
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    ItemOutputMeasures : Planning Item Count,L1 Count,L2 Count,L3 Count, L4 Count,L5 Count,L6 Count,Stat Item Count
    LocationOutputMeasures : Location Type Count,Location Country Count,Reporting Location Count,Stat Location Count,Planning Location Count, Location Region Count
    AccountOutputMeasures : Planning Account Count,Account L1 Count,Account L2 Count,Account L3 Count,Account L4 Count, Stat Account Count
    RegionOutputMeasures : Planning Region Count, Region L1 Count, Region L2 Count,Region L3 Count, Region L4 Count, Stat Region Count
    ChannelOutputMeasures : Channel L1 Count, Channel L2 Count, Planning Channel Count, Stat Channel Count
    PnLOutputMeasures : Planning PnL Count, PnL L1 Count, PnL L2 Count, PnL L3 Count, PnL L4 Count, Stat PnL Count
    DemandDomainOutputMeasures : Planning Demand Domain Count,Demand Domain L1 Count, Demand Domain L2 Count, Demand Domain L3 Count, Demand Domain L4 Count, Stat Demand Domain Count

Input Queries:
    ItemDim : select ([Item].[Planning Item] * [Item].[Transition Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[Item Class] * [Item].[PLC Status] * [Item].[All Item] * [Item].[Item Type] * [Item].[Segmentation LOB]) on row, () on column;

    LocationDim : select ([Location].[All Location] * [Location].[Reporting Location]* [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location]) on row, () on column;

    AccountDim : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column;

    RegionDim : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L4] * [Region].[All Region]) on row, () on column;

    ChannelDim : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column;

    PnLDim : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column;

    DemandDomainDim : select ([Demand Domain].[Transition Demand Domain] * [Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain]) on row, () on column;

    Version : select([Version].[Version Name]);

    StatLevels : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Item Level], Measure.[Region Level], Measure.[Location Level], Measure.[Channel Level], Measure.[Account Level], Measure.[PnL Level], Measure.[Demand Domain Level]}) on column;

Output:
    output

Slice Dimension Attributes : None
"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP051PopulateDimAttributeCount import main

logger = logging.getLogger("o9_logger")


# Function Calls
ItemDim = O9DataLake.get("ItemDim")
LocationDim = O9DataLake.get("LocationDim")
Version = O9DataLake.get("Version")
AccountDim = O9DataLake.get("AccountDim")
PnLDim = O9DataLake.get("PnLDim")
DemandDomainDim = O9DataLake.get("DemandDomainDim")
RegionDim = O9DataLake.get("RegionDim")
ChannelDim = O9DataLake.get("ChannelDim")
StatLevels = O9DataLake.get("StatLevels")

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

output = main(
    ItemDim=ItemDim,
    LocationDim=LocationDim,
    AccountDim=AccountDim,
    DemandDomainDim=DemandDomainDim,
    PnLDim=PnLDim,
    ChannelDim=ChannelDim,
    RegionDim=RegionDim,
    ItemOutputMeasures=ItemOutputMeasures,
    LocationOutputMeasures=LocationOutputMeasures,
    AccountOutputMeasures=AccountOutputMeasures,
    DemandDomainOutputMeasures=DemandDomainOutputMeasures,
    PnLOutputMeasures=PnLOutputMeasures,
    ChannelOutputMeasures=ChannelOutputMeasures,
    RegionOutputMeasures=RegionOutputMeasures,
    StatLevels=StatLevels,
    Version=Version,
    df_keys=df_keys,
)

O9DataLake.put("output", output)
