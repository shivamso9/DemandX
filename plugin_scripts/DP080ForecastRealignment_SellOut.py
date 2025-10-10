"""
Plugin : DP080ForecastRealignment
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:
    HistoryMeasure : Sell Out Actual
    ActualContriOutputMeasure : Realignment Sell Out Actual Contribution %

Input Queries:
    AttributeMapping - Select ([Version].[Version Name]* [Data Object].[Data Object] ) on row,  ({Measure.[Data Object Account Level], Measure.[Data Object Channel Level], Measure.[Data Object Item Level], Measure.[Data Object Location Level], Measure.[Data Object Region Level], Measure.[Data Object PnL Level], Measure.[Data Object Demand Domain Level], Measure.[Data Object Process Order], Measure.[Realign Forecast]}) on column where {Measure.[Realign Forecast] > 0};

    RealignmentRules - Select ([Version].[Version Name] * [Data Object].[Data Object] * [DM Rule].[Rule] * [Sequence].[Sequence] ) on row, ({Measure.[DP From Item Scope], Measure.[DP To Item Scope], Measure.[DP From Account Scope], Measure.[DP To Account Scope], Measure.[DP From Channel Scope], Measure.[DP To Channel Scope], Measure.[DP From Region Scope], Measure.[DP To Region Scope] , Measure.[DP From Location Scope], Measure.[DP To Location Scope],  Measure.[DP From PnL Scope], Measure.[DP To PnL Scope], Measure.[DP From Demand Domain Scope], Measure.[DP To Demand Domain Scope], Measure.[Transition Start Date], Measure.[Transition End Date], Measure.[DP Realignment Percentage],Measure.[DP Conversion Factor], Measure.[Planner Input Realignment Status]}) on column;

    ForecastRaw - Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain]) on row,  ({Measure.[Sell Out Forecast Adjustment 1 FND BB], Measure.[Sell Out Forecast Adjustment 2 FND BB], Measure.[Sell Out Forecast Adjustment 3 FND BB], Measure.[Sell Out Forecast Adjustment 4 FND BB], Measure.[Sell Out Forecast Adjustment 5 FND BB], Measure.[Sell Out Forecast Adjustment 6 FND BB], Measure.[Sell Out Stat Fcst KAF BB New FND BB Raw], Measure.[Sell Out Stat Fcst New FND BB Raw], Measure.[Sell Out Stat Fcst Override FND BB], Measure.[Sell In FND BB Raw]}) on column;

    KeyFigures - Select ([Version].[Version Name] * [Data Object].[Data Object]) on row, ({Measure.[Include in Forecast Realignment], Measure.[Data Object Planner Input]}) on column;

    Actuals - Select ([Version].[Version Name] * [Region].[Region] * [Item].[Item] * [Channel].[Channel] * [Account].[Account] * [Demand Domain].[Demand Domain] * &AllDays.Filter(#.Key <= &CurrentDay.element(0).LeadOffset(-1).Key && #.Key >= &CurrentDay.element(0).LeadOffset(-180).Key).relatedmembers([Partial Week])) on row,  ({Measure.[Sell Out Actual]}) on column include memberproperties {Time.[Partial Week], Key};

    AssortmentFinal - Select ([Version].[Version Name]  * [Item].[Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Location] * [Demand Domain].[Planning Demand Domain] ) on row, ({Measure.[Assortment Final]}) on column;

    IsAssorted - Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] ) on row, ({Measure.[Is Assorted]}) on column;

    AccountMapping - select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]* [Account].[Account]) on row, () on column;

    ChannelMapping - select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] * [Channel].[Channel]) on row, () on column;

    PnLMapping - select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] * [PnL].[PnL]) on row, () on column;

    DemandDomainMapping - select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] *  [Demand Domain].[Planning Demand Domain] *[Demand Domain].[Demand Domain]) on row, () on column;

    LocationMapping - select ([Location].[Planning Location] * [Location].[Location]) on row, () on column;

    TimeDimension - select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month]) on row, () on column include memberproperties {Time.[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key};

    ItemMapping - select ([Item].[Planning Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[All Item] * [Item].[Item]) on row, () on column;

    RegionMapping - select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L3] * [Region].[Region L4] * [Region].[All Region] * [Region].[Region]) on row, () on column;

    ActualContributionParameters - Select ([Version].[Version Name] ) on row,  ({Measure.[Actual Contribution History Bucket], Measure.[Actual Contribution History Period]}) on column;

    CurrentTimePeriod - select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key};

Output Variables:
    RealignedForecast
    Assortment
    IsAssortedOutput
    FlagAndNormalizedValues
    ActualContributionOutput
    RawOverrideOutput

Slice Dimension Attributes:

"""

import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP080ForecastRealignment_SISO import main

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
ForecastRaw = O9DataLake.get("ForecastRaw")
KeyFigures = O9DataLake.get("KeyFigures")
Actuals = O9DataLake.get("Actuals")
AssortmentFinal = O9DataLake.get("AssortmentFinal")
IsAssorted = O9DataLake.get("IsAssorted")
AccountMapping = O9DataLake.get("AccountMapping")
ChannelMapping = O9DataLake.get("ChannelMapping")
PnLMapping = O9DataLake.get("PnLMapping")
DemandDomainMapping = O9DataLake.get("DemandDomainMapping")
LocationMapping = O9DataLake.get("LocationMapping")
TimeDimension = O9DataLake.get("TimeDimension")
ItemMapping = O9DataLake.get("ItemMapping")
RegionMapping = O9DataLake.get("RegionMapping")
ActualContributionParameters = O9DataLake.get("ActualContributionParameters")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")

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


(
    RealignedForecast,
    Assortment,
    IsAssortedOutput,
    FlagAndNormalizedValues,
    ActualContributionOutput,
    RawOverrideOutput,
) = main(
    ActualContriOutputMeasure=ActualContriOutputMeasure,
    HistoryMeasure=HistoryMeasure,
    AttributeMapping=AttributeMapping,
    RealignmentRules=RealignmentRules,
    ForecastRaw=ForecastRaw,
    KeyFigures=KeyFigures,
    Actuals=Actuals,
    AssortmentFinal=AssortmentFinal,
    IsAssorted=IsAssorted,
    AccountMapping=AccountMapping,
    ChannelMapping=ChannelMapping,
    PnLMapping=PnLMapping,
    DemandDomainMapping=DemandDomainMapping,
    LocationMapping=LocationMapping,
    TimeDimension=TimeDimension,
    ItemMapping=ItemMapping,
    RegionMapping=RegionMapping,
    ActualContributionParameters=ActualContributionParameters,
    CurrentTimePeriod=CurrentTimePeriod,
    df_keys=df_keys,
)
O9DataLake.put("RealignedForecast", RealignedForecast)
O9DataLake.put("Assortment", Assortment)
O9DataLake.put("IsAssortedOutput", IsAssortedOutput)
O9DataLake.put("FlagAndNormalizedValues", FlagAndNormalizedValues)
O9DataLake.put("ActualContributionOutput", ActualContributionOutput)
O9DataLake.put("RawOverrideOutput", RawOverrideOutput)
