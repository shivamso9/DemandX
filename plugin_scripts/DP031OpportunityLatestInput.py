"""
Plugin : DP031OpportunityLatestInput
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params: None

Input Queries:
    Opportunity : Select ([Version].[Version Name] * [Time].[Day] * [Account].[Account] * [Channel].[Channel] * [Region].[Region] * [PnL].[PnL] * [Demand Domain].[Demand Domain] * [Location].[Planning Location] * [Item].[Planning Item] * [Opportunity].[Opportunity Line] ) on row, ({Measure.[Opportunity Line Probability Input],Measure.[Opportunity Line Modified Date Input] , Measure.[Opportunity Line Must Win Input], Measure.[Opportunity Line Created Date Input], Measure.[Opportunity Line Leasing Duration Input (in months)], Measure.[Opportunity Line Requested Date Input], Measure.[Opportunity Line Residual Value Input], Measure.[Opportunity Line Revenue Input], Measure.[Opportunity Line Sales Type Input], Measure.[Opportunity Line Stage Input], Measure.[Opportunity Line Units Input]}) on column;

Output Variables:
    output

Slice Dimension Attributes:
    Item.[Planning Item]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP031OpportunityLatestInput import main

logger = logging.getLogger("o9_logger")


# Function Calls
Opportunity = O9DataLake.get("Opportunity")

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

output = main(Opportunity=Opportunity, df_keys=df_keys)

O9DataLake.put("output_df", output)
