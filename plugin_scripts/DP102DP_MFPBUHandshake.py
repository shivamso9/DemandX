"""
Plugin : DP102DP_MFPBUDPHanshake.

Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Input Queries:
   Stat : Select ([Version].[Version Name]  * [Forecast Iteration].[Forecast Iteration] * [Region].[Stat Region] * [Location].[Stat Location] *
  [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Item].[Stat Item] * [Demand Domain].[Stat Demand Domain] * [Time].[Partial Week] * [Account].[Stat Account] ) on row, ({Measure.[Stat Fcst L1]}) on column;

   SellingSeasonAssociation : Select ($$Retail_AP_FD_Product_Location_Setup_NN * $$Retail_AP_FD_Common_NN * [Version].[Version Name]) on row, ({Measure.[Start Week L3 SS], Measure.[End Week L3 SS]}) on column;

   TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter] *  Time.[Year]*  Time.[Week Name]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key} {Time.[Year], Key} {Time.[Week Name], Key};

   CurrentTimePeriod: select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

   TotalSalesUnit: Select ([Item].[L3] *[Channel].[Planning Channel] *  [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [PnL].[Planning PnL] * [Location].[Location Country]* &CWV * [Time].[Partial Week] ) on row,  ({Measure.[Total Sales Unit]}) on column;

Output Variables:
    BUForecastUnt

Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP102DP_MFPBUHandshake import main

logger = logging.getLogger("o9_logger")

# Function Calls
TimeDimension = O9DataLake.get("TimeDimension")
SellingSeasonAssociation = O9DataLake.get("SellingSeasonAssociation")
StatForecast = O9DataLake.get("Stat")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
TotalSalesUnit = O9DataLake.get("TotalSalesUnit")

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


BUForecastUnt = main(
    selling_season=SellingSeasonAssociation,
    stat_forecast=StatForecast,
    time_dimension_df=TimeDimension,
    current_time_period=CurrentTimePeriod,
    total_sales_unit=TotalSalesUnit,
    logger=logger,
    df_keys=df_keys,
)

O9DataLake.put("BUForecastUnt", BUForecastUnt)
