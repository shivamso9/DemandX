"""
Plugin : DP736OverrideStatFcstL0_Pyspark
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:


Input Queries:
    Input (DeltaLake Measure Group) : 320 Forecast

Output Variables:
    Output (DeltaLake Measure Group) : 320 Forecast
"""

import logging

logger = logging.getLogger("o9_logger")

import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake  # type: ignore
from o9Reference.common_utils.o9_memory_utils import _get_memory  # type: ignore

from helpers.DP071ConsensusFcstOverrideRetail import main

Input = O9DataLake.get("Input")


# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.
back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

Output = main(
    Input=Input,
)

O9DataLake.put("Output", Output)
