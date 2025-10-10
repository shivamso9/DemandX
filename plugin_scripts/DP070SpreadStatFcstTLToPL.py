"""
Plugin : DP070SpreadStatFcstTLToPL
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Region.[Planning Region],Item.[Planning Item],PnL.[Planning PnL],Location.[Planning Location],Demand Domain.[Planning Demand Domain],Account.[Planning Account],Channel.[Planning Channel]
    NBucketsinMonths - 12

Input Queries:

    StatFcstFinalProfilePL : Select ([Sequence].[Sequence] * [Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [PnL].[Planning PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Planning Location] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row, ({Measure.[Stat Fcst Final Profile PL], Measure.[Slice Association PL]}) on column where {(Measure.[Slice Association PL] == 1)};

    StatFcstTL : Select ([Sequence].[Sequence] * [Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Transition Item] * [PnL].[Planning PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Planning Location] * [Demand Domain].[Transition Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] ) on row,  ({Measure.[Stat Fcst TL], Measure.[Slice Association PL]}) on column  include memberproperties {Time.[Partial Week], Key} where {(Measure.[Slice Association PL] == 1)};

    CMLFcstTL : Select ([Sequence].[Sequence] * [Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Demand Domain].[Transition Demand Domain] * [Account].[Planning Account] * [Region].[Planning Region] * [Location].[Planning Location] * [Time].[Partial Week] * [Item].[Transition Item] ) on row, ({Measure.[Stat Fcst TL CML Baseline], Measure.[Stat Fcst TL CML External Driver], Measure.[Stat Fcst TL CML Holiday], Measure.[Stat Fcst TL CML Marketing], Measure.[Stat Fcst TL CML Price], Measure.[Stat Fcst TL CML Promo], Measure.[Stat Fcst TL CML Residual], Measure.[Stat Fcst TL CML Weather], Measure.[Slice Association PL]}) on column where {(Measure.[Slice Association PL] == 1)};

    ItemMasterData : select ([Item].[Planning Item] * [Item].[Transition Item]) on row, () on column;

    DemandDomainMasterData : select ([Demand Domain].[Planning Demand Domain] * [Demand Domain].[Transition Demand Domain]) on row, () on column;

    MLDecompositionFlag : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row,  ({Measure.[CML Iteration Decomposition]}) on column;

    StatFcstPLLC : Select ([Sequence].[Sequence] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Planning Region] * [Location].[Planning Location] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] * [Account].[Planning Account] ) on row,  ({Measure.[Stat Fcst PL LC], Measure.[Slice Association PL]}) on column where {(Measure.[Slice Association PL] == 1)};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    PItemMetaData: Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Transition Demand Domain] * [Demand Domain].[Planning Demand Domain] * [Item].[Planning Item] * [Item].[Transition Item]) on row, ({Measure.[Intro Date], Measure.[Disco Date], Measure.[Phase Out Profile], Measure.[Number of Phase Out Buckets], Measure.[Adjust Phase Out Profile], Measure.[Phase In Split %], Measure.[Product Transition Overlap Start Date], Measure.[Transition Type], Measure.[Assortment Phase In]}) on column;

    DefaultProfiles : Select ([Version].[Version Name] * [Lifecycle Time].[Lifecycle Bucket] * [PLC Profile].[PLC Profile] ) on row, ({Measure.[Default Profile]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] *[Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {[Time].[Quarter], Key} {[Time].[Planning Quarter], Key};

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

Output Variables:
    Output
    CMLOutput
    StatFcstPLvsLC_COCC
    Output_volume_loss_flag

Slice Dimension Attributes:
    Sequence.[Sequence]
"""

import logging

from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

import threading

from o9_common_utils.O9DataLake import O9DataLake

from helpers.DP070SpreadStatFcstTLToPL import main

# Function Calls
StatFcstFinalProfilePL = O9DataLake.get("StatFcstFinalProfilePL")
StatFcstTL = O9DataLake.get("StatFcstTL")
CMLFcstTL = O9DataLake.get("CMLFcstTL")
ItemMasterData = O9DataLake.get("ItemMasterData")
DemandDomainMasterData = O9DataLake.get("DemandDomainMasterData")
MLDecompositionFlag = O9DataLake.get("MLDecompositionFlag")
StatFcstPLLC = O9DataLake.get("StatFcstPLLC")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
TimeDimension = O9DataLake.get("TimeDimension")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
PItemMetaData = O9DataLake.get("PItemMetaData")
DefaultProfiles = O9DataLake.get("DefaultProfiles")

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

Output, CMLOutput, StatFcstPLvsLC_COCC, Output_volume_loss_flag = main(
    Grains=Grains,
    StatFcstFinalProfilePL=StatFcstFinalProfilePL,
    StatFcstTL=StatFcstTL,
    CMLFcstTL=CMLFcstTL,
    ItemMasterData=ItemMasterData,
    DemandDomainMasterData=DemandDomainMasterData,
    PItemMetaData=PItemMetaData,
    MLDecompositionFlag=MLDecompositionFlag,
    StatFcstPLLC=StatFcstPLLC,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    StatBucketWeight=StatBucketWeight,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    NBucketsinMonths=NBucketsinMonths,
    DefaultProfiles=DefaultProfiles,
    df_keys=df_keys,
)

O9DataLake.put("Output", Output)
O9DataLake.put("CMLOutput", CMLOutput)
O9DataLake.put("StatFcstPLvsLC_COCC", StatFcstPLvsLC_COCC)
O9DataLake.put("Output_volume_loss_flag", Output_volume_loss_flag)
