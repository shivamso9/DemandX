"""
    Plugin Information:
    --------------------
		Plugin : DP135PublishCannibalizationImpact
		Version : 0.0.0
		Maintained by : dpref@o9solutions.com

    Script Params:
    --------------------


    Input Queries:
    --------------------
        Combinations : Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] ) on row, ({Measure.[Cannib Impact Publish Flag L1]}) on column;

        CurrentDate : select(&CurrentDay)  on row, () on column include memberproperties {Time.[Day], Key};

        CannibIndependenceDate : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Data Object].[Data Object] * [Demand Domain].[Planning Demand Domain] * [Initiative].[Initiative] * [Item].[Planning Item] * [Location].[Planning Location] * [PnL].[Planning PnL] * [Region].[Planning Region] ) on row, ({Measure.[Cannibalization Independence Date Planning Level]}) on column;

        InitiativeLevelStatus	: Select ([Version].[Version Name] * [Data Object].[Data Object] * [Initiative].[Initiative] ) on row, ({Measure.[NPI Initiative Level Status]}) on column;

        PlanningLevelImpact : Select ([Version].[Version Name] * [Initiative].[Initiative] * [Data Object].[Data Object] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] ) on row, ({Measure.[Planning Level Cannibalization Impact]}) on column;

        PublishedImpact :	Select ([Version].[Version Name] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] ) on row,({Measure.[Cannibalization Impact Published]}) on column;

        StatFcst  : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Partial Week] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [PnL].[Planning PnL] * [Region].[Planning Region] ) on row, ({Measure.[Stat Fcst NPI BB]}) on column;

    Output Variables:
    --------------------
        PublishedImpact

    Slice Dimension Attributes:
    -----------------------------
"""

import datetime

# Library imports
import logging
import threading
from functools import reduce

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP135PublishCannibalizationImpact import main

# Pandas Configuration
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

# Load Data from o9DataLake
Combinations = O9DataLake.get("Combinations")
CurrentDate = O9DataLake.get("CurrentDate")
StatFcst = O9DataLake.get("StatFcst")
PublishedImpact = O9DataLake.get("PublishedImpact")
PlanningLevelImpact = O9DataLake.get("PlanningLevelImpact")
InitiativeLevelStatus = O9DataLake.get("InitiativeLevelStatus")
CannibIndependenceDate = O9DataLake.get("CannibIndependenceDate")

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


PublishedImpact_output = main(
    Combinations=Combinations,
    CannibIndependenceDate=CannibIndependenceDate,
    InitiativeLevelStatus=InitiativeLevelStatus,
    PlanningLevelImpact=PlanningLevelImpact,
    StatFcst=StatFcst,
    PublishedImpact=PublishedImpact,
    CurrentDate=CurrentDate,
    df_keys=df_keys,
)

# Save Output Data
O9DataLake.put("PublishedImpact_output", PublishedImpact_output)
