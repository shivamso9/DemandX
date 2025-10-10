"""
    Plugin : DP047HistoryRealignment
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com


Script Params:
OrderedRealignmentInputList - ItemShipToLocationRealignment,ItemLocationRealignment,LocationRealignment,ShipToRealignment

Input:
    ActualRaw - Select ([Version].[Version Name] * [Time].[Day] * [Region].[Region] * [Item].[Item] * [Location].[Location] * [Demand Domain].[Demand Domain] * [Account].[Account] * [Channel].[Channel] * [PnL].[PnL] ) on row, ({Measure.[Actual Raw]}) on column;

    ItemShipToLocationRealignment - Select (FROM.[Item].[Item] * FROM.[Region].[Region] * FROM.[Location].[Location] * TO.[Location].[Location] *TO.[Time].[Day] * [Version].[Version Name]) on row, ({Edge.[930 Item ShipTo Location Realignment].[Item ShipTo Location Realignment Percentage] , Edge.[930 Item ShipTo Location Realignment].[Item ShipTo Location Realignment Percentage Weighted] , Edge.[930 Item ShipTo Location Realignment].[Is Active for Batch]}) on column where {RelationshipType.[930 Item ShipTo Location Realignment]};

    ItemLocationRealignment - Select (FROM.[Item].[Item] * FROM.[Location].[Location] * TO.[Location].[Location] * TO.[Time].[Day] * Version.[Version Name]) on row, ({Edge.[920 Item Location Realignment].[Item Location Realignment Percentage] , Edge.[920 Item Location Realignment].[Item Location Realignment Percentage Weighted] , Edge.[920 Item Location Realignment].[Is Active for Batch]}) on column where {RelationshipType.[920 Item Location Realignment]};

    LocationRealignment - Select (FROM.[Location].[Location] * TO.[Location].[Location] * TO.[Time].[Day] * Version.[Version Name]) on row, ({Edge.[910 Location Realignment].[Location Realignment Percentage] , Edge.[910 Location Realignment].[Location Realignment Percentage Weighted] , Edge.[910 Location Realignment].[Is Active for Batch]}) on column where {RelationshipType.[910 Location Realignment]};

    ShipToRealignment - Select (FROM.[Region].[Region] * TO.[Region].[Region] * TO.[Time].[Day] * Version.[Version Name]) on row, ({Edge.[940 ShipTo Realignment].[ShipTo Realignment Percentage] , Edge.[940 ShipTo Realignment].[ShipTo Realignment Percentage Weighted] , Edge.[940 ShipTo Realignment].[Is Active for Batch]}) on column where {RelationshipType.[940 ShipTo Realignment]};

    CurrentDay - select (&CurrentDay) on row, () on column include memberproperties {[Time].[Day], Key};

    Days - select (Time.[Day]) on row, () on column include memberproperties {[Time].[Day], Key};


Output:

    ActualInput - Select ([Location].[Location] * [Version].[Version Name] * [Sales Domain].[Ship To] * [Time].[Day] * [Item].[Item] ) on row, ({Measure.[Actual Input]}) on column;

Pseudo Code:

    0) Check "ItemShipToLocationRealignment", "ItemLocationRealignment", "LocationRealignment" and "ShipToRealignment" and operate only on the cases which are not blank
        0.1) If all of them are empty. Copy everything which is in "ActualRaw" to the output and change the measure name
        0.2) If "ActualRaw" is blank, make a graceful early exit
    1) Filter "ItemShipToLocationRealignment", "ItemLocationRealignment", "LocationRealignment" and "ShipToRealignment" for rows where "TO.[Time].[Day]" is in history (compare with CurrentDay)
    2) Filter "ItemShipToLocationRealignment", "ItemLocationRealignment", "LocationRealignment" and "ShipToRealignment" for where the edge "[Is Active for Batch]" is "True"
        2.1) Need to have a script param, which if set to "True", we will need to skip the filtering in point (2)
    3) For the remaining rows in "ItemShipToLocationRealignment"
        3.1) Filter and separate out rows from ActualRaw where the Item-ShipTo-Location combinations (in the FROM section) are present in point (3)
        3.2) Based on the value present in "Item ShipTo Location Realignment Percentage Weighted", distribute the corresponding volume to the "TO.[Location].[Location]"
        #ACTUALRAWCARTESIAN
        ACTUALRAW - ACTUALRAWMERGED = DIFF
    4) For the rows remaining in "ActualRaw" post the filter in  step (3.1) and the remaining rows in "ItemLocationRealignment"
        4.1) Filter and separate out rows from "ActualRaw" where the Item-Location combinations (in the FROM section) are present in "ItemLocationRealignment"
        4.2) Based on the value present in "Item Location Realignment Percentage Weighted", distribute the corresponding volume to the "TO.[Location].[Location]" = PROCESSDIFF
    5) For the rows remaining in "ActualRaw" post the filter in step (4.1) and the remaining rows in "LocationRealignment"
        5.1) Filter and separate out rows from "ActualRaw" where the Locations (in the FROM section) are present in "LocationRealignment"
        5.2) Based on the value present in "Location Realignment Percentage Weighted", distribute the corresponding volume to the "TO.[Location].[Location]"
    6) For the rows remaining in "ActualRaw" post the filter in step (5.1) and the remaining rows in "ShipToRealignment"
        6.1) Filter and separate out rows from "ActualRaw" where the ShipTos (in the FROM section) are present in "ShipToRealignment"
        6.2) Based on the value present in "ShipTo Realignment Percentage Weighted", distribute the corresponding volume to the "TO.[Location].[Location]"
    7) The outputs of steps 3,4,5 and 6 along with whatever is pending in "ActualRaw" to be appended
    8) The name of the output measure is "Actual Input" and it has the same grain as in the measure in "ActualRaw"

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP047HistoryRealignment import main

logger = logging.getLogger("o9_logger")

# Function Calls
ActualRaw = O9DataLake.get("ActualRaw")
ItemShipToLocationRealignment = O9DataLake.get("ItemShipToLocationRealignment")
ItemLocationRealignment = O9DataLake.get("ItemLocationRealignment")
LocationRealignment = O9DataLake.get("LocationRealignment")
ShipToRealignment = O9DataLake.get("ShipToRealignment")
CurrentDay = O9DataLake.get("CurrentDay")
Days = O9DataLake.get("Days")

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


ActualInput = main(
    ActualRaw=ActualRaw,
    ItemShipToLocationRealignment=ItemShipToLocationRealignment,
    ItemLocationRealignment=ItemLocationRealignment,
    LocationRealignment=LocationRealignment,
    ShipToRealignment=ShipToRealignment,
    OrderedRealignmentInputList=OrderedRealignmentInputList,
    CurrentDay=CurrentDay,
    Days=Days,
    df_keys=df_keys,
)

O9DataLake.put("ActualInput", ActualInput)
