"""
    Plugin : DP104DP_MFPLPHandshake
    Version : 2025.08.00
    Maintained by : dpref@o9solutions.com

    Description : This plugin processes the StatFcstL1 and SellingSeasonAssociation dataframes to create a LP Forecast Unit dataframe.

    Input Queries:
        StatFcstL1 : Select ([Version].[Version Name] * [Item].[Stat Item] * [Channel].[Stat Channel] * [Time].[Week] * [Time].[Planning Month] * [Location].[Stat Location] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Stat Fcst L1]}) on column;
        SellingSeasonAssociation : Select ([Item].[L4] * [Location].[Location] * $$Retail_AP_FD_Common_NN * [Version].[Version Name]) on row, ({Measure.[Start Week L3 SS], Measure.[End Week L3 SS]}) on column;
        CurrentTime : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};
        TimeMaster : select ([Time].[Day] * Time.[Partial Week] * Time.[Week] * Time.[Month] * Time.[Year] * Time.[Planning Month] * Time.[Quarter] * Time.[Planning Quarter] *Time.[Week Name]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    Output Variables:
        Output : LPForecastUnit

    Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP104DP_MFPLPHandshake import main

logger = logging.getLogger("o9_logger")

# Function Calls
StatFcstL1 = O9DataLake.get("StatFcstL1")
SellingSeasonAssociation = O9DataLake.get("SellingSeasonAssociation")
CurrentTime = O9DataLake.get("CurrentTime")
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

LPForecastUnit = main(
    stat_fcst=StatFcstL1,
    selling_season=SellingSeasonAssociation,
    time_df=TimeMaster,
    current_time_period=CurrentTime,
    logger=logger,
    df_keys=df_keys,
)

O9DataLake.put("LPForecastUnit", LPForecastUnit)
