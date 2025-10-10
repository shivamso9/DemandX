"""
Plugin : DP040AttachRateSystem
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    MovingAvgPeriod - 4
    ReadFromHive - False
    MultiprocessingNumCores - 4

Input Queries:
    edge - Select (FROM.[Item].[Planning Item] * TO.[Item].[Planning Item] * TO.[Time].[Week]  * Version.[Version Name]) on row, ({Edge.[810 Attach Rate Planning].[Attach Rate System]}) on column  where {RelationshipType.[810 Attach Rate Planning]};
    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};
    TimeDimension - select ([Time].[Week] * [Time].[Partial Week]) on row, () on column include memberproperties {Time.[Week], Key} {Time.[Partial Week], Key};

Output Variable:
    MovingAvg - Graph

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
attach_rate = O9DataLake.get("edge")
TimeDimension = O9DataLake.get("TimeDimension")
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

from helpers.DP040AttachRateSystem import main

MovingAvg = main(
    MovingAvgPeriod=MovingAvgPeriod,
    attach_rate=attach_rate,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    df_keys=df_keys,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
)

O9DataLake.put("MovingAvg", MovingAvg)
