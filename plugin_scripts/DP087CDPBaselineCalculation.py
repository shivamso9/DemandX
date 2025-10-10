"""This module calculates the consensus baseline Forecast.

Plugin : DP087CDPBaselineCalculation.
Version : 2025.08.00
Maintained by : dpref@o9solutions.com


Input Queries:
    ProcessOrder : Select ([Version].[Version Name] * [Data Object].[Data Object] ) on row, ({Measure.[Data Object Account Level], Measure.[Data Object Channel Level], Measure.[Data Object Demand Domain Level], Measure.[Data Object Item Level], Measure.[Data Object Location Level], Measure.[Data Object PnL Level], Measure.[Data Object Region Level], Measure.[Data Object Process Order]}) on column where {[Data Object].[Data Object Type].[Consensus Baseline]};

    AssortmentFlag: Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] ) on row,    ({Measure.[Assortment Collab DP]}) on column where {Measure.[Assortment Collab DP] == 1.0};

    SegmentationInput: (Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [Location].[Planning Location] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] ) on row,  ({Measure.[Assortment Final],Measure.[Sales Fcst],Measure.[Marketing Fcst],Measure.[Stat Fcst L0],Measure.[Stat Fcst L0 Final],Measure.[NPI Fcst Collab BB]}) on column where{&ConsensusForecastBuckets}).filter(Measure.[Assortment Collab DP]==1);

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

    RuleScope: Select ([Version].[Version Name] * [Data Object].[Data Object]  * [DM Rule].[Rule] ) on row, ({Measure.[DP Forecast Type] , Measure.[DP Location Scope UI] , Measure.[DP Item Scope UI] , Measure.[DP Account Scope UI] , Measure.[DP Channel Scope UI] , Measure.[DP Region Scope UI] , Measure.[DP Demand Domain Scope UI] , Measure.[DP PnL Scope UI] }) on column  where {coalesce(Measure.[DP Exclude Flag], false) != true , [Data Object].[Data Object Type].[Consensus Baseline]};

    CandidateFcst: (Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * &ConsensusForecastBuckets * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [PnL].[Planning PnL] * [Region].[Planning Region] ) on row,  ({Measure.[Consensus Fcst],    Measure.[Stat Fcst L0], Measure.[Stat Fcst L0 Final],    Measure.[NPI Fcst Collab BB],    Measure.[Sales Fcst],    Measure.[Marketing Fcst],    Measure.[Reference1 Fcst],Measure.[Reference2 Fcst],Measure.[Reference3 Fcst]}) on column).filter(Measure.[Assortment Collab DP]==1);

    TimeMaster : Select (&ConsensusForecastBuckets* [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter]* [Time].[Planning Quarter] * [Time].[Year] * [Time].[Planning Year]) on row, () on column include memberproperties {[Time].[Partial Week], Key} {[Time].[Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key} {[Time].[Quarter], Key} { [Time].[Planning Quarter] , Key} { [Time].[Year] , Key} { [Time].[Planning Year], Key} where{&ConsensusForecastBuckets.relatedmembers([Day])};


    Touchless: Select ([Version].[Version Name] * [Region].[Planning Region]  * [Location].[Planning Location] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item]  * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row,    ({Measure.[Is Already Touchless]}) on column;


    CurrentTimePeriod: Select (&CurrentPartialWeek * [Time].[Week] * [Time].[Day] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {[Time].[Day], Key} {Time.[Month], Key} {Time.[Planning Month], Key};


    Horizon: Select ([Version].[Version Name] * [Data Validation].[Data Validation Type].[Consensus Baseline].relatedmembers([Data Validation]) ) on row,  ({Measure.[Consensus Horizon Bucket], Measure.[Consensus Horizon Period], Measure.[Horizon Process Order], Measure.[Is Last Horizon]}) on column;

    HorizonRule : Select ([Version].[Version Name] * [Data Object].[Data Object Type].[Consensus Baseline].relatedmembers([Data Object]) * [Data Validation].[Data Validation Type].[Consensus Baseline].relatedmembers([Data Validation]) * [DM Rule].[Rule] ) on row,  ({Measure.[Horizon Candidate Fcst], Measure.[Horizon Period by Rule], Measure.[Horizon Rule Association], Measure.[Outside Horizon Candidate Fcst]}) on column;

    ReferenceFcst : Select ([Version].[Version Name] * [Cluster].[Cluster] ) on row,  ({Measure.[Reference Fcst for Cluster]}) on column;

    CurrentPlanningCycle : Select(&CurrentPlanningCycle)on row, () on column include memberproperties {[Planning Cycle].[Planning Cycle], Key};

    LastDay : select (&ConsensusForecastBuckets.relatedmembers([Day]).last())on row, () on column include memberproperties {[Time].[Day], Key}{[Time].[Partial Week], Key};

    CannibalizationImpactFlag : Select (&CWV * [Item].[Planning Item] * [Channel].[Planning Channel] * &ConsensusForecastBuckets * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [PnL].[Planning PnL] * [Region].[Planning Region] ) on row, ({Measure.[Cannibalization Impact Flag]}) on column;

     CannibalizationImpactCandidate : Select ([Version].[Version Name] ) on row, ({Measure.[Candidate to Include Cannibalization Impact]}) on column;

Output Variables:

    ConsensusBaselineForecast : 320 Forecast
    PlanningRuleHorizonMapping : 319 Candidate Fcst by Rule

Slice Dimension Attributes:

"""

import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP087CDPBaselineCalculation import main

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

# Function Calls

# Function Calls
ProcessOrder = O9DataLake.get("ProcessOrder")
RuleScope = O9DataLake.get("RuleScope")
AssortmentFlag = O9DataLake.get("AssortmentFlag")
SegmentationInput = O9DataLake.get("SegmentationInput")

AccountMaster = O9DataLake.get("AccountMaster")
ChannelMaster = O9DataLake.get("ChannelMaster")
DemandDomainMaster = O9DataLake.get("DemandDomainMaster")
ItemMaster = O9DataLake.get("ItemMaster")
LocationMaster = O9DataLake.get("LocationMaster")
PnLMaster = O9DataLake.get("PnLMaster")
RegionMaster = O9DataLake.get("RegionMaster")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
TimeMaster = O9DataLake.get("TimeMaster")

CandidateFcst = O9DataLake.get("CandidateFcst")
Horizon = O9DataLake.get("Horizon")
HorizonRule = O9DataLake.get("HorizonRule")
CurrentPlanningCycle = O9DataLake.get("CurrentPlanningCycle")
ReferenceFcst = O9DataLake.get("ReferenceFcst")
LastDay = O9DataLake.get("LastDay")
CannibalizationImpactFlag = O9DataLake.get("CannibalizationImpactFlag")
CannibalizationImpactCandidate = O9DataLake.get("CannibalizationImpactCandidate")

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

ConsensusBaselineForecast, PlanningRuleHorizonMapping = main(
    ProcessOrder=ProcessOrder,
    RuleScope=RuleScope,
    AssortmentFlag=AssortmentFlag,
    SegmentationInput=SegmentationInput,
    AccountMaster=AccountMaster,
    ChannelMaster=ChannelMaster,
    DemandDomainMaster=DemandDomainMaster,
    ItemMaster=ItemMaster,
    LocationMaster=LocationMaster,
    PnLMaster=PnLMaster,
    RegionMaster=RegionMaster,
    CandidateFcst=CandidateFcst,
    HorizonRule=HorizonRule,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeMaster=TimeMaster,
    Horizon=Horizon,
    ReferenceFcst=ReferenceFcst,
    CannibalizationImpactFlag=CannibalizationImpactFlag,
    CurrentPlanningCycle=CurrentPlanningCycle,
    LastDay=LastDay,
    CannibalizationImpactCandidate=CannibalizationImpactCandidate,
    df_keys=df_keys,
)

O9DataLake.put("Consensus Baseline Forecast", ConsensusBaselineForecast)
O9DataLake.put("Planning Rule Horizon Mapping", PlanningRuleHorizonMapping)
