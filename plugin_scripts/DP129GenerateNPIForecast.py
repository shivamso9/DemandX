"""
    Plugin Information:
    --------------------
        Plugin : DP129GenerateNPIForecast
        Version : 0.0.0
        Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------
        NPIStartDateFormat: None
        NPIEndDateFormat: None
        TimeMasterKeyFormat: None

    Input Queries:
    --------------------
        SelectedCombinations: Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] ) on row, ({Measure.[Generate System NPI Fcst Assortment L0]}) on column;

        LikeItemBasedParameters: Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] ) on row, ({Measure.[NPI Association L0], Measure.[Start Date L0], Measure.[End Date L0], Measure.[NPI Forecast Generation Method L0], Measure.[User Defined NPI Profile L0], Measure.[NPI Ramp Up Bucket L0], Measure.[NPI Ramp Up Period L0], Measure.[Initial Build L0], Measure.[Scaling Factor L0], Measure.[System Suggested Ramp Up Volume L0], Measure.[User Defined Ramp Up Volume L0]}) on column;

        ManualParameters: Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] * [Item].[NPI Item] * [Account].[NPI Account] * [Channel].[NPI Channel] * [Region].[NPI Region] * [PnL].[NPI PnL] * [Demand Domain].[NPI Demand Domain] * [Location].[NPI Location] ) on row, ({Measure.[NPI Association L0], Measure.[Start Date L0], Measure.[End Date L0], Measure.[NPI Forecast Generation Method L0], Measure.[User Defined NPI Profile L0], Measure.[NPI Bucket L0], Measure.[User Defined NPI Period L0], Measure.[Initial Build L0], Measure.[User Defined Total Volume L0]}) on column;

        LaunchProfile: Select ([Version].[Version Name] * [Item].[NPI Item] * [Data Object].[Data Object] * [Channel].[NPI Channel] * [Lifecycle Time].[Lifecycle Bucket] * [Location].[NPI Location] * [Account].[NPI Account] * [Demand Domain].[NPI Demand Domain] * [PnL].[NPI PnL] * [Initiative].[Initiative] * [Region].[NPI Region] ) on row, ({Measure.[Ramp Up Profile Final L0]}) on column;

        PLCProfile: Select ([PLC Profile].[PLC Profile] * [Version].[Version Name] * [Lifecycle Time].[Lifecycle Bucket] ) on row, ({Measure.[Default Profile]}) on column include memberproperties {[PLC Profile].[PLC Profile], [PLC Time Bucket]} {[Lifecycle Time].[Lifecycle Bucket], Key};

        LikeItemFcst: Select ([Version].[Version Name] * [Item].[NPI Item] * [Data Object].[Data Object] * [Channel].[NPI Channel] * [Time].[Partial Week] * [Location].[NPI Location] * [Account].[NPI Account] * [Demand Domain].[NPI Demand Domain] * [PnL].[NPI PnL] * [Initiative].[Initiative] * [Region].[NPI Region] ) on row, ({Measure.[Like Item Fcst L0]}) on column;

        NumDays: Select ([Version].[Version Name] * [Time].[Partial Week] ) on row, ({Measure.[Num Days]}) on column;

        TimeMaster: Select ([Time].[Day] * [Time].[Week] * [Time].[Partial Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {[Time].[Week], Key} {[Time].[Partial Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

    Output Variables:
    --------------------
        NPIForecast

    Slice Dimension Attributes:

"""

# Library imports
import logging
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP129GenerateNPIForecast import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

# Load Data from o9DataLake
SelectedCombinations = O9DataLake.get("SelectedCombinations")
Parameters = O9DataLake.get("Parameters")
LaunchProfile = O9DataLake.get("LaunchProfile")
PLCProfile = O9DataLake.get("PLCProfile")
LikeItemFcst = O9DataLake.get("LikeItemFcst")
NumDays = O9DataLake.get("NumDays")
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


NPIForecast = main(
    # Params
    NPIStartDateFormat=NPIStartDateFormat,
    NPIEndDateFormat=NPIEndDateFormat,
    TimeMasterKeyFormat=TimeMasterKeyFormat,
    # Data
    SelectedCombinations=SelectedCombinations,
    Parameters=Parameters,
    LaunchProfile=LaunchProfile,
    PLCProfile=PLCProfile,
    LikeItemFcst=LikeItemFcst,
    NumDays=NumDays,
    TimeMaster=TimeMaster,
    df_keys=df_keys,
)


# Save Output Data
O9DataLake.put("NPIForecast", NPIForecast)
