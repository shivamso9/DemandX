"""
Plugin : DP022PopulateLifecycleActual
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    ReadFromHive - False
    Grains - Item.[Item],Location.[Location],Channel.[Planning Channel],Account.[Planning Account],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Region.[Planning Region]
    TimeLevel - Time.[Day]

Input Queries:
    TimeDimension : Select ([Time].[Day]) on row, () on column include memberproperties {[Time].[Day], Key};

    Input (primary) : Select ([Item].[Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain]* [Location].[Location] * [Time].[Day] * [Version].[Version Name]) on row, ({Measure.[Actual],Measure.[Billing],Measure.[Backorders],Measure.[Orders],Measure.[Shipments]}) on column;

Output Variables:
    output

Slice Dimension Attributes:

"""

import logging

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


logger = logging.getLogger("o9_logger")

import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

# Function Calls
Input = O9DataLake.get("Input")
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

from helpers.DP022PopulateLifecycleActual import main

output = main(
    input_df=Input,
    TimeDimension=TimeDimension,
    TimeLevel=TimeLevel,
    Grains=Grains,
    ReadFromHive=ReadFromHive,
    df_keys=df_keys,
)
O9DataLake.put("output", output)
