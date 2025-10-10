"""
Plugin : DP049PopulateNPIFcst
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    Grains : Item.[Planning Item],Channel.[Planning Channel],Account.[Planning Account],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Region.[Planning Region],Location.[Planning Location]

    ReadFromHive : False

Input Queries:
    forecastData : Select ([Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Version].[Version Name] * [Time].[Partial Week] * [Item].[Planning Item] ) on row, ({Measure.[Like Item Fcst]}) on column;

    Parameters : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Item].[Planning Item]) on row, ({Measure.[Intro Date],Measure.[Disco Date],Measure.[User Defined Ramp Up Period],Measure.[User Defined Ramp Up Volume],Measure.[Scaling Factor],Measure.[Initial Build], Measure.[NPI Profile], Measure.[NPI Profile Bucket]}) on column;

    TimeDimension : Select ([Time].[Day] * [Time].[Week] * [Time].[Partial Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {[Time].[Week], Key} {[Time].[Partial Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

    NumDaysInput : Select ([Version].[Version Name] * [Time].[Partial Week] ) on row,  ({Measure.[Num Days]}) on column;

    DefaultProfile : Select ([PLC Profile].[PLC Profile] * [Version].[Version Name] * [Lifecycle Time].[Lifecycle Bucket] ) on row, ({Measure.[Default Profile]}) on column include memberproperties {[PLC Profile].[PLC Profile], [PLC Time Bucket]} {[Lifecycle Time].[Lifecycle Bucket], Key};

Output Variables:
    NPIForecast

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

# Function Calls
forecastData = O9DataLake.get("forecastData")
TimeDimension = O9DataLake.get("TimeDimension")
Parameters = O9DataLake.get("Parameters")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
DefaultProfile = O9DataLake.get("DefaultProfile")

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

from helpers.DP049PopulateNPIFcst import main

NPIForecast = main(
    forecast_data=forecastData,
    parameter_data=Parameters,
    stat_bucket_weight=NumDaysInput,
    Grains=Grains,
    df_keys=df_keys,
    TimeDimension=TimeDimension,
    DefaultProfile=DefaultProfile,
)

O9DataLake.put("NPIForecast", NPIForecast)
