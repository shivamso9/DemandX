"""
Plugin : DP043SellInCalc
Version : 0.0.0
Maintained by : pmm_algocoe@o9solutions.com

Script Params:
    Grains - Item.[Planning Item],Channel.[Planning Channel],Account.[Planning Account],Demand Domain.[Planning Demand Domain],Region.[Planning Region]
    LeadTimeGrains=Region.[Planning Region]


Input Queries:

InTransitInventory - Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Week] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Region].[Planning Region] ) on row,  ({Measure.[Sell Out], Measure.[In Transit Inventory], Measure.[In Transit Inventory Agg], Measure.[Total Demand], Measure.[Ch INV], Measure.[Sell In Override]}) on column;

IncludePastInTransitInventory - Select ([Version].[Version Name]) on row, ({Measure.[Include Past In Transit Inventory]}) on column;

LeadTime - Select ([Version].[Version Name] * [Region].[Planning Region]) on row, ({Measure.[Lead Time]}) on column;

ConsiderLeadTime - Select ([Version].[Version Name]) on row, ({Measure.[Consider Lead Time]}) on column;

CurrentWeek - select(&CurrentWeek) on row, () on column include memberproperties {[Time].[Week], Key};

TimeDimension - Select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {[Time].[Partial Week], Key} {[Time].[Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

Output Variables:
    ChINVL1
    ChINV
    TotalSupply
    SellInCalc
    SellInCalcLT
    SellInAM

Slice Dimension Attributes:

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP043SellInCalc import main

logger = logging.getLogger("o9_logger")

# Function Calls
InTransitInventory = O9DataLake.get("InTransitInventory")
IncludePastInTransitInventory = O9DataLake.get("IncludePastInTransitInventory")
ConsiderLeadTime = O9DataLake.get("ConsiderLeadTime")
LeadTime = O9DataLake.get("LeadTime")
CurrentWeek = O9DataLake.get("CurrentWeek")
TimeDimension = O9DataLake.get("TimeDimension")
# DONE : Please provide a meaningful name - done

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

SellOutSellIn = main(
    grains=Grains,
    Lead_Time_Grains=LeadTimeGrains,
    InTransitInventory=InTransitInventory,
    IncludePastInTransitInventory=IncludePastInTransitInventory,
    LeadTime=LeadTime,
    CurrentWeek=CurrentWeek,
    TimeDimension=TimeDimension,
    ConsiderLeadTime=ConsiderLeadTime,
    df_keys=df_keys,
)

O9DataLake.put("SellOutSellIn", SellOutSellIn)
