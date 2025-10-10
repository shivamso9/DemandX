"""
Plugin : DP060PopulateFcstOutsidePLCHorizonFlag
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    OutputMeasure : Fcst Outside PLC Horizon Flag
    Grains : Region.[Planning Region],Item.[Planning Item],PnL.[Planning PnL],Location.[Planning Location],Demand Domain.[Planning Demand Domain],Account.[Planning Account],Channel.[Planning Channel]

Input Queries:
    ConsensusFcst : Select ([Version].[Version Name] * &AllPlanningItem * &AllPlanningAccount * &AllPlanningChannel * &AllPlanningRegion * &AllPlanningDemandDomain * &AllPlanningPnL * &AllPlanningLocation * &PLCForecastBuckets ) on row, ({Measure.[Consensus Fcst]}) on column;

    IntroDiscDates - Select ([Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] *  [Location].[Planning Location]  * [Version].[Version Name] * [Item].[Planning Item] ) on row, ({Measure.[Intro Date], Measure.[Disco Date]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

    CurrentDay : Select (&CurrentDay) on row, () on column include memberproperties {[Time].[Day], Key};

Output Variables:
    Output

Slice Dimension Attributes: None

"""

import logging

from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

import numpy as np
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
ConsensusFcst = O9DataLake.get("ConsensusFcst")
IntroDiscDates = O9DataLake.get("IntroDiscDates")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentDay = O9DataLake.get("CurrentDay")

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

from helpers.DP060PopulateFcstOutsidePLCHorizonFlag import main

Output = main(
    ConsensusFcst=ConsensusFcst,
    IntroDiscDates=IntroDiscDates,
    TimeDimension=TimeDimension,
    CurrentDay=CurrentDay,
    Grains=Grains,
    OutputMeasure=OutputMeasure,
    df_keys=df_keys,
)

O9DataLake.put("Output", Output)
