"""
Plugin : DP012SetupAlgoAssociation
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]

Input Queries:
    Parameters : Select ([Version].[Version Name] * [Stat Algorithm].[Stat Algorithm] * [Stat Parameter].[Stat Parameter] ) on row,  ({Measure.[Stat Algorithm Parameter Association]}) on column;

    OverrideFlags : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Region].[Stat Region] * [Item].[Stat Item] * [Location].[Stat Location] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] * [PnL].[Stat PnL] ) on row,  ({Measure.[Rule Override Flag], Measure.[Algorithm List Override Flag]}) on column;

    AlgoList : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Version].[Version Name] * [Location].[Stat Location] * [Item].[Stat Item] * [Account].[Stat Account] * [PnL].[Stat PnL] * [Region].[Stat Region] * [Channel].[Stat Channel] * [Demand Domain].[Stat Demand Domain] ) on row,  ({Measure.[Assigned Algorithm List], Measure.[Slice Association Stat]}) on column where {~(Measure.[Assigned Algorithm List] contains "No Forecast"), ~isnull(Measure.[Assigned Algorithm List]), (Measure.[Slice Association Stat] == 1)};

Output Variables:
    AlgoAssociation
    ParameterAssociation

Slice Dimension Attributes: [Sequence].[Sequence]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP012SetupAlgoAssociation import main

logger = logging.getLogger("o9_logger")

# Function Calls
Parameters = O9DataLake.get("Parameters")
OverrideFlags = O9DataLake.get("OverrideFlags")
AlgoList = O9DataLake.get("AlgoList")

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

AlgoAssociation, ParameterAssociation = main(
    Parameters=Parameters,
    OverrideFlags=OverrideFlags,
    AlgoList=AlgoList,
    Grains=Grains,
    df_keys=df_keys,
)

O9DataLake.put("AlgoAssociation", AlgoAssociation)
O9DataLake.put("ParameterAssociation", ParameterAssociation)
