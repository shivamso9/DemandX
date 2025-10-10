"""
    Plugin : DP013OverridePlannerRuleLocal
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com

    Script Params:
        Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]

    Input Queries:
        GlobalAlgoList : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Stat Rule].[Stat Rule] ) on row,  ({Measure.[Planner Algorithm List]}) on column include memberproperties {[Stat Rule].[Stat Rule], [System Algorithm List]};

        RuleNos : Select ([Sequence].[Sequence] * [Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Item].[Stat Item] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] * [PnL].[Stat PnL] ) on row, ({Measure.[Assigned Rule], Measure.[Slice Association Stat]}) on column where {(Measure.[Rule Override Flag]==True), ~isnull(Measure.[Actual]), (Measure.[Slice Association Stat] == 1)};

    Output Variables:
        LocalAlgoList

    Slice Dimension Attributes: [Sequence].[Sequence]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP013OverridePlannerRuleLocal import main

logger = logging.getLogger("o9_logger")


RuleNos = O9DataLake.get("RuleNos")
GlobalAlgoList = O9DataLake.get("GlobalAlgoList")
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

LocalAlgoList = main(
    Grains=Grains,
    GlobalAlgoList=GlobalAlgoList,
    RuleNos=RuleNos,
    df_keys=df_keys,
)
O9DataLake.put("LocalAlgoList", LocalAlgoList)
