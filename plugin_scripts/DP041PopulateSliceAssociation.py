"""
Plugin : DP041PopulateSliceAssociation
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    StatGrains : Location.[Stat Location],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Item.[Stat Item]
    NumIntersectionsInOneSlice : 1100
    TransitionGrains : Location.[Planning Location],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Transition Demand Domain],Item.[Transition Item]
    PlanningGrains : Location.[Planning Location],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Item.[Planning Item]
    StatPartitionBy : None
    TranstionPartitionBy : Location.[Stat Location],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Item.[Stat Item]
    PlanningPartitionBy : Demand Domain.[Transition Demand Domain],Item.[Transition Item]
    DefaultMapping : Item-All Planning Item,Account-All Planning Account,Location-All Planning Location,Region-All Planning Region,Channel-All Planning Channel,Demand Domain-All Planning Demand Domain,PnL-All Planning PnL

Input Queries:
    StatActuals - Select ( [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Item].[Stat Item] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location]  *{Measure.[Stat Actual]});

    ForecastIterationSelection : Select ([Version].[Version Name] * [Region].[Planning Region] * [Item].[Planning Item] * [Item].[Transition Item] * [PnL].[Planning PnL] * [Forecast Iteration].[Forecast Iteration Type] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] ) on row,  ({Measure.[Forecast Iteration Selection]}) on column;

    ForecastLevelData : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Item Level], Measure.[Account Level], Measure.[Channel Level], Measure.[PnL Level], Measure.[Region Level], Measure.[Demand Domain Level], Measure.[Location Level]}) on column;

    ItemMasterData : select ([Item].[Planning Item] * [Item].[Transition Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[Item Class] * [Item].[PLC Status] * [Item].[All Item] * [Item].[Item Type] * [Item].[Segmentation LOB]) on row, () on column  include_nullmembers;

    RegionMasterData : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region]) on row, () on column include_nullmembers;

    AccountMasterData : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column include_nullmembers;

    ChannelMasterData : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column include_nullmembers;

    PnLMasterData : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column include_nullmembers;

    DemandDomainMasterData : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Transition Demand Domain] *  [Demand Domain].[Planning Demand Domain]) on row, () on column include memberproperties {[Demand Domain].[All Demand Domain], DisplayName} {[Demand Domain].[Demand Domain L1], DisplayName} {[Demand Domain].[Demand Domain L2], DisplayName} {[Demand Domain].[Demand Domain L3], DisplayName} {[Demand Domain].[Demand Domain L4], DisplayName} {[Demand Domain].[Planning Demand Domain], DisplayName} include_nullmembers;

    LocationMasterData :  select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location]) on row, () on column include_nullmembers;

    AssortmentStat : Select ([Version].[Version Name] * [Region].[Planning Region] * [Location].[Planning Location] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Item].[Transition Item] * [Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain] * [Account].[Planning Account] ) on row,  ({Measure.[Assortment Stat]}) on column;

    ForecastIterationMasterData : Select (Version.[Version Name] * [Forecast Iteration].[Forecast Iteration Type] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Iteration Type Input Stream]}) on column;

    AssortmentCustom : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row,  ({Measure.[Sell Out Actual]}) on column include memberproperties {Time.[Partial Week],Key};

Output Tables:
    StatOutput
    TransitionOutput
    PlanningOutput

------ PSEUDOCODE ---------

Objective : Given a dataframe of intersections populate a 'Sequence' column such that
            equal number of intersections are present in all groups or sequences.

1. Reset index to obtain row number for every row
2. For every row, take modulus of row number by NumBuckets (and add 1) to assign values to the slice sequence. Modulus can be zero, hence we add 1 to ensure sequence starts from digit 1
3. Sort condition ensures that splitting is even across all the groups present
4. Add Slice Association Stat value (1.0) to all intersections
5. Create a function that does the above steps but without sorting and repeat them for TransitionHistory and PlanningActual
6. Filter the grain columns and new slice association column and return as Output, TransitionOutput and PlanningOutput

"""

from o9_common_utils.O9DataLake import O9DataLake

from helpers.DP041PopulateSliceAssociation import main

# Function Calls

StatActuals = O9DataLake.get("StatActuals")
ForecastIterationSelection = O9DataLake.get("ForecastIterationSelection")
ForecastLevelData = O9DataLake.get("ForecastLevelData")
ItemMasterData = O9DataLake.get("ItemMasterData")
RegionMasterData = O9DataLake.get("RegionMasterData")
AccountMasterData = O9DataLake.get("AccountMasterData")
ChannelMasterData = O9DataLake.get("ChannelMasterData")
PnLMasterData = O9DataLake.get("PnLMasterData")
DemandDomainMasterData = O9DataLake.get("DemandDomainMasterData")
LocationMasterData = O9DataLake.get("LocationMasterData")
ForecastIterationMasterData = O9DataLake.get("ForecastIterationMasterData")
AssortmentStat = O9DataLake.get("AssortmentStat")
AssortmentCustom = O9DataLake.get("AssortmentCustom")


DefaultMapping_list = DefaultMapping.strip().split(",")
default_mapping = {}
for mapping in DefaultMapping_list:
    key, value = mapping.split("-")
    default_mapping[key] = value


StatOutput, TransitionOutput, PlanningOutput = main(
    StatGrains=StatGrains,
    NumIntersectionsInOneSlice=NumIntersectionsInOneSlice,
    StatActuals=StatActuals,
    ForecastIterationSelection=ForecastIterationSelection,
    TransitionGrains=TransitionGrains,
    PlanningGrains=PlanningGrains,
    StatPartitionBy=StatPartitionBy,
    TransitionPartitionBy=TransitionPartitionBy,
    PlanningPartitionBy=PlanningPartitionBy,
    ForecastLevelData=ForecastLevelData,
    ItemMasterData=ItemMasterData,
    RegionMasterData=RegionMasterData,
    AccountMasterData=AccountMasterData,
    ChannelMasterData=ChannelMasterData,
    PnLMasterData=PnLMasterData,
    DemandDomainMasterData=DemandDomainMasterData,
    LocationMasterData=LocationMasterData,
    ForecastIterationMasterData=ForecastIterationMasterData,
    default_mapping=default_mapping,
    AssortmentStat=AssortmentStat,
    AssortmentCustom=AssortmentCustom,
)

O9DataLake.put("StatOutput", StatOutput)
O9DataLake.put("TransitionOutput", TransitionOutput)
O9DataLake.put("PlanningOutput", PlanningOutput)
