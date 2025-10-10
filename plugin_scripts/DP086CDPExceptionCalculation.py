"""
Plugin : ExceptionsCDP
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    1. Lag = 1
    2. DPExceptionCalculationWindow = 12,3,3 (CoCC, Accuracy, Bias)
    3. DPExceptionTolerance = 0.1,0.75,0.1 (CoCC, Accuracy, Bias)
    4. MinToleranceFreq = 2,2,2 (CoCC, Accuracy, Bias)
    5. FVA = -0.05 (Accuracy)

Input Queries:
    ProcessOrder: Select ([Version].[Version Name] * [Data Object].[Data Object] ) on row,
    ({Measure.[Data Object Account Level], Measure.[Data Object Channel Level], Measure.[Data Object Demand Domain Level], Measure.[Data Object Item Level], Measure.[Data Object Location Level],
    Measure.[Data Object PnL Level], Measure.[Data Object Region Level], Measure.[Data Object Process Order]}) on column where {[Data Object].[Data Object Type].[Exceptions CDP]};
    ExceptionParams: Select ([Version].[Version Name] * [Data Object].[Data Object]  * [DM Rule].[Rule] ) on row,
    ({Measure.[DP Exception Type] , Measure.[DP Lag Scope] , Measure.[DP Cluster Scope] , Measure.[DP Location Scope UI] , Measure.[DP Item Scope UI] , Measure.[DP Account Scope UI] , Measure.[DP Channel Scope UI] , Measure.[DP Region Scope UI] , Measure.[DP Demand Domain Scope UI],
    Measure.[DP PnL Scope UI] , Measure.[DP Exception Min Tolerance Freq] , Measure.[DP Exception Calculation Window] , Measure.[DP Exception Tolerance]  , Measure.[DP Exception FVA Tolerance]}) on column  where {coalesce(Measure.[DP Exclude Flag], false) != true };
    AssortmentFlag: Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] ) on row,
    ({Measure.[Assortment Collab DP], Measure.[Assortment Exception DP], Measure.[Is Mature Flag]}) on column where {Measure.[Assortment Collab DP] == 1.0};
    SegmentationFlag: Select ([Version].[Version Name].[Plugin_Version] * [Account].[Planning Account] * [Item].[Planning Item] * [Cluster].[Cluster] ) on row,
    ({Measure.[CDP Segmentation Flag]}) on column;
    AccountMaster: Select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account] * [Account].[Account]) on row,
    () on column include memberproperties {[Account].[Account L1], [Account L1$DisplayName]}{[Account].[Account L2], [Account L2$DisplayName]}{[Account].[Account L3], [Account L3$DisplayName]}{[Account].[Account L4],
    [Account L4$DisplayName]}{[Account].[All Account], [All Account$DisplayName]}{[Account].[Planning Account], [Planning Account$DisplayName]}{[Account].[Account], [Account$DisplayName]};
    ChannelMaster: Select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] * [Channel].[All Channel] * [Channel].[Channel]) on row,
    () on column include memberproperties {[Channel].[Channel L1], [Channel L1$DisplayName]}{[Channel].[Channel L2], [Channel L2$DisplayName]}{[Channel].[Planning Channel],
    [Planning Channel$DisplayName]}{[Channel].[All Channel], [All Channel$DisplayName]}{[Channel].[Channel], [Channel$DisplayName]};
    DemandDomainMaster: Select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *
    [Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Demand Domain]) on row,
    () on column include memberproperties {[Demand Domain].[All Demand Domain], [All Demand Domain$DisplayName]}{[Demand Domain].[Demand Domain L1],
    [Demand Domain L1$DisplayName]}{[Demand Domain].[Demand Domain L2], [Demand Domain L2$DisplayName]}{[Demand Domain].[Demand Domain L3],
    [Demand Domain L3$DisplayName]}{[Demand Domain].[Demand Domain L4], [Demand Domain L4$DisplayName]}{[Demand Domain].[Planning Demand Domain],
    [Planning Demand Domain$DisplayName]}{[Demand Domain].[Demand Domain], [Demand Domain$DisplayName]};
    ItemMaster: Select ([Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[All Item] * [Item].[Item]) on row,
    () on column include memberproperties {[Item].[Planning Item], [Planning Item$DisplayName]}{[Item].[L1], [L1$DisplayName]}{[Item].[L2],
    [L2$DisplayName]}{[Item].[L3], [L3$DisplayName]}{[Item].[L4], [L4$DisplayName]}{[Item].[L5], [L5$DisplayName]}{[Item].[L6], [L6$DisplayName]}{[Item].[All Item], [All Item$DisplayName]}{[Item].[Item], [Item$DisplayName]};
    LocationMaster: Select ([Location].[All Location] *[Location].[Location Type] * [Location].[Location Country] * [Location].[Location Region] * [Location].[Planning Location] * [Location].[Location]) on row, () on column;
    PnLMaster: Select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] * [PnL].[PnL]) on row,
    () on column include memberproperties {[PnL].[All PnL], [All PnL$DisplayName]}{[PnL].[Planning PnL], [Planning PnL$DisplayName]}{[PnL].[PnL L1], [PnL L1$DisplayName]}
    {[PnL].[PnL L2], [PnL L2$DisplayName]}{[PnL].[PnL L3], [PnL L3$DisplayName]}{[PnL].[PnL L4], [PnL L4$DisplayName]}{[PnL].[PnL], [PnL$DisplayName]};
    RegionMaster: Select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region] * [Region].[Region]) on row,
    () on column include memberproperties {[Region].[Planning Region], [Planning Region$DisplayName]}{[Region].[Region L1], [Region L1$DisplayName]}{[Region].[Region L2],
    [Region L2$DisplayName]}{[Region].[Region L3], [Region L3$DisplayName]}{[Region].[Region L4], [Region L4$DisplayName]}{[Region].[All Region], [All Region$DisplayName]}{[Region].[Region], [Region$DisplayName]};
    ActualL0: Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [Location].[Planning Location]
    * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Time].[Month]) on row,
    ({Measure.[Actual L0]}) on column;
    SystemAndConsensusFcst: Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [Location].[Planning Location]
    * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * Lag.[Lag] *[Time].[Month]) on row,  ({Measure.[System Fcst Mature M Lag], Measure.[Consensus Fcst Mature M Lag]}) on column;
    StatAndConsensusFcst: Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [Location].[Planning Location]
    * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Time].[Month]) on row,
    ({Measure.[Consensus Fcst],Measure.[Consensus Fcst LC],Measure.[Stat Fcst L0 Final],Measure.[Stat Fcst L0 Final LC]}) on column;
    Touchless: Select ([Version].[Version Name] * [Region].[Planning Region]  * [Location].[Planning Location] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item]
    * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row,
    ({Measure.[Is Already Touchless]}) on column;
    CurrentTimePeriod: Select (&CurrentPartialWeek * [Time].[Week] * [Time].[Day] * [Time].[Month] * [Time].[Planning Month]) on row,
    () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {[Time].[Day], Key} {Time.[Month], Key} {Time.[Planning Month], Key};
    TimeMaster: Select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row,
    () on column include memberproperties {[Time].[Partial Week], Key} {[Time].[Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

Output Variables:
    ExceptionParamsOP: 359 DP Exceptions Parameter at Planning Level
    ExceptionWithLagOP: 357 DP Exceptions with Lag

Slice Dimension Attributes: None
"""

import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP086CDPExceptionCalculation import main

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

# Function Calls
ProcessOrder = O9DataLake.get("ProcessOrder")
ExceptionParams = O9DataLake.get("ExceptionParams")
AssortmentFlag = O9DataLake.get("AssortmentFlag")
SegmentationFlag = O9DataLake.get("SegmentationFlag")

AccountMaster = O9DataLake.get("AccountMaster")
ChannelMaster = O9DataLake.get("ChannelMaster")
DemandDomainMaster = O9DataLake.get("DemandDomainMaster")
ItemMaster = O9DataLake.get("ItemMaster")
LocationMaster = O9DataLake.get("LocationMaster")
PnLMaster = O9DataLake.get("PnLMaster")
RegionMaster = O9DataLake.get("RegionMaster")

ActualL0 = O9DataLake.get("ActualL0")
SystemAndConsensusFcst = O9DataLake.get("SystemAndConsensusFcst")
StatAndConsensusFcst = O9DataLake.get("StatAndConsensusFcst")
Touchless = O9DataLake.get("Touchless")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
TimeMaster = O9DataLake.get("TimeMaster")

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

ExceptionParamsOP, ExceptionWithLagOP = main(
    Lag=Lag,
    DPExceptionCalculationWindow=DPExceptionCalculationWindow,
    DPExceptionTolerance=DPExceptionTolerance,
    MinToleranceFreq=MinToleranceFreq,
    FVA=FVA,
    ProcessOrder=ProcessOrder,
    ExceptionParams=ExceptionParams,
    AssortmentFlag=AssortmentFlag,
    SegmentationFlag=SegmentationFlag,
    AccountMaster=AccountMaster,
    ChannelMaster=ChannelMaster,
    DemandDomainMaster=DemandDomainMaster,
    ItemMaster=ItemMaster,
    LocationMaster=LocationMaster,
    PnLMaster=PnLMaster,
    RegionMaster=RegionMaster,
    ActualL0=ActualL0,
    SystemAndConsensusFcst=SystemAndConsensusFcst,
    StatAndConsensusFcst=StatAndConsensusFcst,
    Touchless=Touchless,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeMaster=TimeMaster,
    df_keys=df_keys,
)

O9DataLake.put("ExceptionParamsOP", ExceptionParamsOP)
O9DataLake.put("ExceptionWithLagOP", ExceptionWithLagOP)
