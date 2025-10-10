"""
Plugin : DP063GenerateStatActual
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    PlanningGrains - Channel.[Planning Channel],Demand Domain.[Planning Demand Domain],Region.[Planning Region],Account.[Planning Account],PnL.[Planning PnL],Location.[Location],Item.[Planning Item]
    InputTables - Actual,SellOutActual

Input Queries:
    Actual : Select ([Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Version].[Version Name] * [Region].[Planning Region] * [Account].[Planning Account] * [PnL].[Planning PnL] * [Location].[Location] * [Time].[Partial Week] * [Item].[Planning Item]) on row,  ({Measure.[Actual]}) on column include memberproperties {Time.[Partial Week],Key};

    ForecastLevelData : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row, ({Measure.[Item Level], Measure.[Account Level], Measure.[Channel Level], Measure.[PnL Level], Measure.[Region Level], Measure.[Demand Domain Level], Measure.[Location Level]}) on column;

    ItemMasterData : select ([Item].[Planning Item] *  [Item].[Transition Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4] * [Item].[L5] * [Item].[L6] * [Item].[Item Class] * [Item].[PLC Status] * [Item].[All Item]  * [Item].[Segmentation LOB]) on row, () on column include memberproperties {Item.[Planning Item], DisplayName} {Item.[L1], DisplayName} {Item.[L2], DisplayName} {Item.[L3], DisplayName} {Item.[L4], DisplayName} {Item.[L5], DisplayName} {Item.[L6], DisplayName} {Item.[Item Class], DisplayName} {Item.[PLC Status], DisplayName} {Item.[All Item], DisplayName} {Item.[Transition Item], DisplayName} {Item.[Segmentation LOB], DisplayName} include_nullmembers;

    RegionMasterData : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region]) on row, () on column include memberproperties {[Region].[Planning Region], DisplayName} {[Region].[Region L1], DisplayName} {[Region].[Region L2], DisplayName}  {[Region].[Region L3], DisplayName} {[Region].[Region L4], DisplayName} {[Region].[All Region], DisplayName} include_nullmembers;

    AccountMasterData : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column include memberproperties {[Account].[Account L1], DisplayName} {[Account].[Account L2], DisplayName} {[Account].[Account L3], DisplayName} {[Account].[Account L4], DisplayName} {[Account].[All Account], DisplayName} {[Account].[Planning Account], DisplayName} include_nullmembers;

    ChannelMasterData : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column include memberproperties {[Channel].[Channel L1], DisplayName} {[Channel].[Channel L2], DisplayName} {[Channel].[Planning Channel], DisplayName} {[Channel].[All Channel], DisplayName} include_nullmembers;

    PnLMasterData : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column include memberproperties {[PnL].[All PnL], DisplayName} {[PnL].[Planning PnL], DisplayName} {[PnL].[PnL L1], DisplayName} {[PnL].[PnL L2], DisplayName} {[PnL].[PnL L3], DisplayName} {[PnL].[PnL L4], DisplayName} include_nullmembers;

    DemandDomainMasterData : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain]) on row, () on column include memberproperties {[Demand Domain].[All Demand Domain], DisplayName} {[Demand Domain].[Demand Domain L1], DisplayName} {[Demand Domain].[Demand Domain L2], DisplayName} {[Demand Domain].[Demand Domain L3], DisplayName} {[Demand Domain].[Demand Domain L4], DisplayName} {[Demand Domain].[Planning Demand Domain], DisplayName} {[Demand Domain].[Transition Demand Domain], DisplayName} include_nullmembers;

    LocationMasterData : select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location] * [Location].[Reporting Location]) on row, () on column include memberproperties {Location.[Planning Location], DisplayName} {Location.[Location Country], DisplayName} {Location.[Location Region], DisplayName} {Location.[Location Type], DisplayName} {Location.[All Location], DisplayName} {Location.[Location], DisplayName }  {[Location].[Reporting Location], DisplayName} include_nullmembers;

    ForecastIterationMasterData : Select (Version.[Version Name] * [Forecast Iteration].[Forecast Iteration Type] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Iteration Type Input Stream]}) on column;

    SellOutActual : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row,  ({Measure.[Sell Out Actual]}) on column include memberproperties {Time.[Partial Week],Key};

Output Variables:
    Output

Slice Dimension Attributes:None
"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP063GenerateStatActual_MFP import main

logger = logging.getLogger("o9_logger")
# Function Calls
ForecastLevelData = O9DataLake.get("ForecastLevelData")
ItemMasterData = O9DataLake.get("ItemMasterData")
RegionMasterData = O9DataLake.get("RegionMasterData")
AccountMasterData = O9DataLake.get("AccountMasterData")
ChannelMasterData = O9DataLake.get("ChannelMasterData")
PnLMasterData = O9DataLake.get("PnLMasterData")
DemandDomainMasterData = O9DataLake.get("DemandDomainMasterData")
ForecastIterationMasterData = O9DataLake.get("ForecastIterationMasterData")
LocationMasterData = O9DataLake.get("LocationMasterData")

# configurable inputs
try:
    CombinedActual = {}
    AllInputTables = InputTables.strip().split(",")
    for i in AllInputTables:
        Input = O9DataLake.get(i)
        if Input is None:
            logger.exception(f"Input '{i}' is None, please recheck the table name...")
            continue
        CombinedActual[i] = Input
        logger.warning(f"Reading input {i}...")
except Exception as e:
    logger.exception(f"Error while reading inputs\nError:-\n{e}")


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

Output, ActualL0 = main(
    Actual=CombinedActual,
    ForecastLevelData=ForecastLevelData,
    ItemMasterData=ItemMasterData,
    RegionMasterData=RegionMasterData,
    AccountMasterData=AccountMasterData,
    ChannelMasterData=ChannelMasterData,
    PnLMasterData=PnLMasterData,
    DemandDomainMasterData=DemandDomainMasterData,
    LocationMasterData=LocationMasterData,
    ForecastIterationMasterData=ForecastIterationMasterData,
    Grains=Grains,
    PlanningGrains=PlanningGrains,
    df_keys=df_keys,
)

O9DataLake.put("Output", Output)
O9DataLake.put("ActualL0", ActualL0)
