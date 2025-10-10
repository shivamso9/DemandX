"""
Plugin : DP070FlexibleLevelLocationSplit
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    HistoryMeasure - Actual
    Grains - [Region].[Planning Region],[Demand Domain].[Planning Demand Domain],[Account].[Planning Account],[Channel].[Planning Channel],[PnL].[Planning PnL],Item.[Item],Location.[Location]
    TimeLevel - Time.[Partial Week]
    ReadFromHive - False
    MultiprocessingNumCores - 4

Input Queries:
    Actual : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Item].[Item] * [Time].[Partial Week]) on row, ({Measure.[Actual]}) on column;

    CurrentTimePeriod : Select (&CurrentDay * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column;

    TimeDimension : Select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Partial Week], Key} {[Time].[Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

    LocationSplitParameters : Select ([Version].[Version Name]) on row, ({Measure.[Location Split History Period], Measure.[Location Split History Time Bucket]}) on column;

    LocationSplitPercentInputNormalized : Select ([Region].[Planning Region] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Time].[Partial Week] * [Item].[Item] * [Location].[Location] ) on row,  ({Measure.[Location Split % Input Normalized], Measure.[Location Split Assortment]}) on column where {Measure.[Location Split Assortment] == 1.0};

    ForecastBucket : Select (&DPSPForecastBuckets) on row, () on column include memberproperties {[Time].[Partial Week], Key};

    LocationSplitMethod : Select ([Region].[Planning Region] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Item].[Item] ) on row, ({Measure.[Location Split Method Final]}) on column;

    ItemConsensusFcst : Select ([Version].[Version Name] * [Item].[Item] * [Region].[Planning Region] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Time].[Partial Week] ) on row, ({Measure.[Item Consensus Fcst]}) on column;

    AssortmentFlag : Select ([Region].[Planning Region] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Version].[Version Name] * [Item].[Item] * [Location].[Location] ) on row, ({Measure.[Assortment Final]}) on column where {Measure.[Assortment Final] > 0.0};

    ItemAttribute - Select ([Item].[Item]*[Item].[Planning Item]*[Item].[Transition Item]*[Item].[Stat Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]*[Item].[Item Stage]*[Item].[Item Type]*[Item].[PLC Status]*[Item].[Item Class]*[Item].[Transition Group]*[Item].[A1]*[Item].[A2]*[Item].[A3]*[Item].[A4]*[Item].[A5]*[Item].[A6]*[Item].[A7]*[Item].[A8]*[Item].[A9]*[Item].[A10]*[Item].[A11]*[Item].[A12]*[Item].[A13]*[Item].[A14]*[Item].[A15]*[Item].[All Item]) on row,() on column include memberproperties {[Item].[Item],[Item Intro Date]}{[Item].[Item],[Item Disc Date]}{[Item].[Item],[Item Status]}{[Item].[Item],[Is New Item]} INCLUDE_NULLMEMBERS;

    DPRule - Select ([Version].[Version Name] * [Data Object].[Data Object] * [DM Rule].[Rule] ) on row,  ({Measure.[DP Account Level], Measure.[DP Account Scope], Measure.[DP Channel Level],  Measure.[DP Channel Scope], Measure.[DP Demand Domain Level],  Measure.[DP Demand Domain Scope], Measure.[DP Item Level], Measure.[DP Item Scope],  Measure.[DP PnL Level], Measure.[DP PnL Scope], Measure.[DP Region Level], Measure.[DP Region Scope], Measure.[DP Rule Created By], Measure.[DP Rule Created Date], Measure.[DP Rule Method], Measure.[DP Rule Sequence], Measure.[DP Rule Flag]}) on column;

    Select ([Version].[Version Name] * &DPSPForecastBuckets * [Location].[Location] * [Data Object].[Data Object] * [DM Rule].[Rule] ) on row,  ({Measure.[Location Split Flexible Assortment], Measure.[Location Split Flexible Input], Measure.[Location Split Flexible Normalized]}) on column where {Measure.[Location Split Flexible Assortment]>0.0};

    ChannelAttribute - select ([Channel].[Planning Channel]*[Channel].[Channel L1]*[Channel].[Channel L2] * [Channel].[All Channel] ) on row, () on column;

    RegionAttribute - select ([Region].[Planning Region]*[Region].[Region L1]*[Region].[Region L2]*[Region].[Region L3]* [Region].[Region L4] * [Region].[All Region] ) on row, () on column;

    PnLAttribute - select ([PnL].[Planning PnL]*[PnL].[PnL L1]*[PnL].[PnL L2]*[PnL].[PnL L3]* [PnL].[PnL L4] * [PnL].[All PnL] ) on row, () on column;

    DemandDomainAttribute - select ([Demand Domain].[Planning Demand Domain]*[Demand Domain].[Demand Domain L1]*[Demand Domain].[Demand Domain L2]*[Demand Domain].[Demand Domain L3]* [Demand Domain].[Demand Domain L4] * [Demand Domain].[All Demand Domain] ) on row, () on column;

    AccountAttribute - select ([Account].[Planning Account]*[Account].[Account L1]*[Account].[Account L2]*[Account].[Account L3]* [Account].[Account L4] * [Account].[All Account] ) on row, () on column;

Output Variables:
    Assortment_location
    FlexibleOutput
    finalfcst_col
    SplitFlexible
    SplitFinal
    RuleSelected
    FlexibleMethod



"""

import logging
import threading
from threading import Thread

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


# Function Calls
# Read Inputs
logger.info("Reading data from o9DataLake ...")
AccountAttribute = O9DataLake.get("AccountAttribute")
Actual = O9DataLake.get("Actual")
AssortmentFlag = O9DataLake.get("AssortmentFlag")
ChannelAttribute = O9DataLake.get("ChannelAttribute")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
DemandDomainAttribute = O9DataLake.get("DemandDomainAttribute")
DPRule = O9DataLake.get("DPRule")
FlexibleInput = O9DataLake.get("FlexibleInput")
ForecastBucket = O9DataLake.get("ForecastBucket")
ItemAttribute = O9DataLake.get("ItemAttribute")
ItemConsensusFcst = O9DataLake.get("ItemConsensusFcst")
LocationSplitParameters = O9DataLake.get("LocationSplitParameters")
PnLAttribute = O9DataLake.get("PnLAttribute")
RegionAttribute = O9DataLake.get("RegionAttribute")
TimeDimension = O9DataLake.get("TimeDimension")

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

from helpers.DP070FlexibleLevelLocationSplit import main

AssortmentSplitLocation, FlexibleInpputDFOutput, FinalFcstOutput = main(
    Grains=Grains,
    AccountAttribute=AccountAttribute,
    Actual=Actual,
    AssortmentFlag=AssortmentFlag,
    ChannelAttribute=ChannelAttribute,
    CurrentTimePeriod=CurrentTimePeriod,
    DemandDomainAttribute=DemandDomainAttribute,
    DPRule=DPRule,
    FlexibleInput=FlexibleInput,
    ForecastBucket=ForecastBucket,
    ItemAttribute=ItemAttribute,
    ItemConsensusFcst=ItemConsensusFcst,
    LocationSplitParameters=LocationSplitParameters,
    PnLAttribute=PnLAttribute,
    RegionAttribute=RegionAttribute,
    TimeDimension=TimeDimension,
    HistoryMeasure=HistoryMeasure,
    df_keys=df_keys,
)

O9DataLake.put("AssortmentSplitLocation", AssortmentSplitLocation)
O9DataLake.put("FlexibleInpputDFOutput", FlexibleInpputDFOutput)
O9DataLake.put("FinalFcstOutput", FinalFcstOutput)
