"""
Plugin : DP054InputChecks
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:

    HistoryPeriodWeeks - 156
    HolidaySpikeLag - 2
    HolidaySpikeLead - 2
    HolidayDipLag - 2
    HolidayDipLead - 2
    PromoSpikeLag - 0
    PromoSpikeLead - 0
    PromoDipLag - 0
    PromoDipLead - 2
    StockoutPeriod - 2
    StockoutThreshold - 0.1
    Grains - Item.[Stat Item],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location]
    MINIMUM_PROMINENCE - 10
    weekWindowLength - 26
    monthWindowLength - 6
    quarterWindowLength - 2
    MultiprocessingNumCores - 4

Input Queries:

    Actuals : Select ([Forecast Iteration].[Forecast Iteration] * [Sequence].[Sequence] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Location].[Stat Location] * [Version].[Version Name] * [Item].[Stat Item] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Actual], Measure.[Slice Association Stat]}) on column where {~isnull(Measure.[Stat Actual]), (Measure.[Slice Association Stat] == 1)};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type].relatedmembers([Forecast Iteration]) ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    HolidayData :  Select ([Version].[Version Name] * [Region].[Planning Region] *  [Time].[Partial Week] ) on row,  ({Measure.[Is Holiday]}) on column;

    PromotionData : Select ([Version].[Version Name] *  [Region].[Stat Region] * [Demand Domain].[Stat Demand Domain] * [Account].[Stat Account] * [Channel].[Stat Channel] * [PnL].[Stat PnL] * [Time].[Partial Week] * [Item].[Stat Item] ) on row,  ({Measure.[Promo Days]}) on column;

    FlagsOutCols : Select ([Version].[Version Name] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Time].[Partial Week] * [Item].[Stat Item] ) on row,  ({Measure.[Holiday Spikes Flag], Measure.[Holiday Dips Flag], Measure.[Promo Spikes Flag], Measure.[Promo Dips Flag], Measure.[Other Spikes Flag], Measure.[Other Dips Flag], Measure.[Holiday Spikes Offset], Measure.[Holiday Dips Offset], Measure.[Promo Spikes Offset], Measure.[Promo Dips Offset], Measure.[Potential Stockout Period], Measure.[Holiday Flag], Measure.[Promo Flag], Measure.[Spike or Dip], Measure.[Spikes Flag], Measure.[Dips Flag]}) on column limit 1;

    OutputColumnNames : Select ([Version].[Version Name] * [Location].[Stat Location] * [Account].[Stat Account] * [Channel].[Stat Channel] * [Region].[Stat Region] * [PnL].[Stat PnL] * [Demand Domain].[Stat Demand Domain] * [Item].[Stat Item] ) on row,  ({Measure.[Holiday Spikes],Measure.[Promo Spikes],Measure.[Holiday Dips],Measure.[Promo Dips],Measure.[Other Spikes],Measure.[Other Dips],Measure.[Potential Stockout Flag],Measure.[Other Spikes and Dips],Measure.[Total Spikes],Measure.[Total Dips],Measure.[Total Spikes and Dips]}) on column limit 1;

    ForecastLevelData : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Item Level], Measure.[Account Level], Measure.[Channel Level], Measure.[PnL Level], Measure.[Region Level], Measure.[Demand Domain Level]}) on column;

    RegionMasterData : select ([Region].[Planning Region] * [Region].[Region L1] * [Region].[Region L2] * [Region].[Region L4] * [Region].[All Region]) on row, () on column include memberproperties {[Region].[Planning Region], DisplayName} {[Region].[Region L1], DisplayName} {[Region].[Region L2], DisplayName} {[Region].[Region L4], DisplayName} {[Region].[All Region], DisplayName};

    ItemMasterData : select ([Item].[Planning Item] * [Item].[Transition Item] * [Item].[L1] * [Item].[L2] * [Item].[L3] * [Item].[L4]* [Item].[L5]* [Item].[L6] * [Item].[Item Class] * [Item].[PLC Status] * [Item].[All Item] * [Item].[Item Type] * [Item].[Segmentation LOB]) on row, () on column;

    AccountMasterData : select ([Account].[Account L1] * [Account].[Account L2] * [Account].[Account L3] * [Account].[Account L4] * [Account].[All Account] * [Account].[Planning Account]) on row, () on column;

    ChannelMasterData : select ([Channel].[Channel L1] * [Channel].[Channel L2] * [Channel].[Planning Channel] *[Channel].[All Channel] ) on row, () on column;

    PnLMasterData : select ([PnL].[All PnL] * [PnL].[Planning PnL] * [PnL].[PnL L1] * [PnL].[PnL L2] * [PnL].[PnL L3] * [PnL].[PnL L4] ) on row, () on column;

    DemandDomainMasterData : select ([Demand Domain].[All Demand Domain] * [Demand Domain].[Demand Domain L1] * [Demand Domain].[Demand Domain L2] *[Demand Domain].[Demand Domain L3] * [Demand Domain].[Demand Domain L4] * [Demand Domain].[Transition Demand Domain] *  [Demand Domain].[Planning Demand Domain]) on row, () on column;

    LocationMasterData : select ([Location].[All Location] * [Location].[Location Type] * [Location].[Location] * [Location].[Location Region] *  [Location].[Location Country] * [Location].[Planning Location]) on row, () on column;

    SellOutOffset : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Offset Period]}) on column;

Output Variables:

    Flags
    Values

Slice Dimension Attributes:
    Sequence.[Sequence]

Pseudocode:

    1. Read input data that includes:
        1.1. Actuals
        1.2. Time Master
        1.3. Last Time Bucket
        1.4. Forecast Generation Level
        1.5. Holiday data
        1.6. Promotions data
        1.7. Disaggregation weights

    2. Preprocess input data:
        2.1. Merge Actuals with History and Promotions to make one input table. If History or Promotions is empty populate with zeros.
        2.2. Filter Time Master and remove unnecessary grains

    3. Process the Actuals data for each Item, Location, Account, Demand Domain, Region, Channel, PnL intersections in the following sequence:
        3.1. Fill in the missing intersections in the time horizon by merging each table with the time master and then aggregating to the Forecast Generation Level.
        3.2. To populate the "Potential Stockout Period" flag, check for StockoutPeriod or more consecutive buckets where the Actual value is 0 (where StockoutPeriod is a script parameter that denotes minimum stockout period)
        3.3. Find Spikes and Dips in the Actuals timeseries. The Window Length and minimum prominence are script parameters. The Spikes flag, Dips flag and the Spike or Dip Flag are set accordingly.
        3.4. The Holiday and Promotion flags are shifted by the Lead and Lag parameters for each driver. A union of the original series and the offsetted series is then used to check for alignment with the Spikes and Dips flags. The Holiday Spikes, Holiday Dips, Promo Spikes and Promo Dips flags are set accordingly. The spikes and dips that don't align with any of the Holiday or Promotion flags are marked as Other Spikes and Other Dips.
        3.5. The Driver Spike Offset and Driver Dip Offset flags are used to indicate which of the offsetted series have aligned with the spikes or dips.

    4. The processed data, returned as the Flags dataframe. Before it is written to the LS, it needs to be disaggregated to the FIRST Partial Week time grain:

    5. The Flags table is aggregated to get the Values table.

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP054InputChecks import main

logger = logging.getLogger("o9_logger")


# Function Calls
Actuals = O9DataLake.get("Actuals")
TimeDimension = O9DataLake.get("TimeDimension")
HolidayData = O9DataLake.get("HolidayData")
PromotionData = O9DataLake.get("PromotionData")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
FlagsOutCols = O9DataLake.get("FlagsOutCols")
OutputColumnNames = O9DataLake.get("OutputColumnNames")
ForecastLevelData = O9DataLake.get("ForecastLevelData")
RegionMasterData = O9DataLake.get("RegionMasterData")
ItemMasterData = O9DataLake.get("ItemMasterData")
AccountMasterData = O9DataLake.get("AccountMasterData")
ChannelMasterData = O9DataLake.get("ChannelMasterData")
PnLMasterData = O9DataLake.get("PnLMasterData")
DemandDomainMasterData = O9DataLake.get("DemandDomainMasterData")
LocationMasterData = O9DataLake.get("LocationMasterData")
SellOutOffset = O9DataLake.get("SellOutOffset")

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

(Flags, Values) = main(
    Actuals=Actuals,
    TimeDimension=TimeDimension,
    CurrentTimePeriod=CurrentTimePeriod,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    HolidayData=HolidayData,
    PromotionData=PromotionData,
    FlagsOutCols=FlagsOutCols,
    OutputColumnNames=OutputColumnNames,
    HistoryPeriodWeeks=HistoryPeriodWeeks,
    HolidaySpikeLag=HolidaySpikeLag,
    HolidaySpikeLead=HolidaySpikeLead,
    HolidayDipLag=HolidayDipLag,
    HolidayDipLead=HolidayDipLead,
    PromoSpikeLag=PromoSpikeLag,
    PromoSpikeLead=PromoSpikeLead,
    PromoDipLag=PromoDipLag,
    PromoDipLead=PromoDipLead,
    StockoutPeriod=StockoutPeriod,
    StockoutThreshold=StockoutThreshold,
    Grains=Grains,
    MINIMUM_PROMINENCE=MINIMUM_PROMINENCE,
    weekWindowLength=weekWindowLength,
    monthWindowLength=monthWindowLength,
    quarterWindowLength=quarterWindowLength,
    df_keys=df_keys,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    ForecastLevelData=ForecastLevelData,
    RegionMasterData=RegionMasterData,
    ItemMasterData=ItemMasterData,
    AccountMasterData=AccountMasterData,
    ChannelMasterData=ChannelMasterData,
    PnLMasterData=PnLMasterData,
    DemandDomainMasterData=DemandDomainMasterData,
    LocationMasterData=LocationMasterData,
    SellOutOffset=SellOutOffset,
)

O9DataLake.put("Flags", Flags)
O9DataLake.put("Values", Values)
