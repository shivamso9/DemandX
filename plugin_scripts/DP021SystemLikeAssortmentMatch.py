"""
    Plugin : DP021SystemLikeAssortmentMatch
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com

Script Params:

    FeatureLevel : Item.[L1]

    ReadFromHive : False

    numerical_cols : None

Input Queries:
    ItemAttribute :  Select ([Item].[Item]*[Item].[Planning Item]*[Item].[Transition Item]*[Item].[Stat Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]*[Item].[Item Stage]*[Item].[Item Type]*[Item].[PLC Status]*[Item].[Item Class]*[Item].[Transition Group]*[Item].[A1]*[Item].[A2]*[Item].[A3]*[Item].[A4]*[Item].[A5]*[Item].[A6]*[Item].[A7]*[Item].[A8]*[Item].[A9]*[Item].[A10]*[Item].[A11]*[Item].[A12]*[Item].[A13]*[Item].[A14]*[Item].[A15]*[Item].[All Item]) on row,() on column include memberproperties {[Item].[Item],[Item Intro Date]}{[Item].[Item],[Item Disc Date]}{[Item].[Item],[Item Status]}{[Item].[Item],[Is New Item]} INCLUDE_NULLMEMBERS;

    PlanningItemAttribute : Select ([Item].[Planning Item]) on row, () on column include memberproperties {[Item].[Planning Item],[Planning Item Intro Date]}{[Item].[Planning Item],[Planning Item Disc Date]}{[Item].[Planning Item],[Planning Item Locale]}{[Item].[Planning Item],[Base UOM]}{[Item].[Planning Item],[UOM Type]}{[Item].[Planning Item],[Container Size]}{[Item].[Planning Item],[Package Size]} INCLUDE_NULLMEMBERS;

    SearchSpace : Select ([Version].[Version Name] * [DM Rule].[Rule] ) on row,  ({Measure.[Like Account Search Space L0], Measure.[Like Channel Search Space L0], Measure.[Like Demand Domain Search Space L0], Measure.[Like Item Search Space L0], Measure.[Like Location Search Space L0], Measure.[Like PnL Search Space L0], Measure.[Like Region Search Space L0]}) on column;

    Parameters : Select ([Version].[Version Name] * [DM Rule].[Rule] ) on row,  ({Measure.[Like Assortment Search History Measure], Measure.[Like Assortment Min History Period], Measure.[Like Assortment Launch Window], Measure.[Num Like Assortment]}) on column;

    Sales : Select ([Version].[Version Name] * [Time].[Day] * [Region].[Planning Region] * [Item].[Item] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] ) on row, ({Measure.[Like Item Actual]}) on column;

    AttributeWeights : Select ([Version].[Version Name] * [Item].[L1] * [DM Rule].[Rule] * [Item Feature].[Item Feature] ) on row,  ({Measure.[System Feature Weight L0], Measure.[User Feature Weight L0]}) on column;

    AssortmentMapping : Select ([Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Version].[Version Name] * [Location].[Planning Location]) on row, ({Measure.[Is Assorted]}) on column ;

    ConsensusFcst : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * &ConsensusForecastBuckets * [Item].[Planning Item] * [Location].[Planning Location] ) on row, ({Measure.[Stat Fcst NPI BB]}) on column;

    GenerateSystemLikeItemMatchAssortment : Select ([Version].[Version Name] * [Region].[Planning Region] * [Location].[Planning Location] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row, ({Measure.[Generate System Like Item Match Assortment]}) on column;

    AccountAttribute : Select ([Account].[All Account] * [Account].[Account L4] * [Account].[Account L3] * [Account].[Account L2] * [Account].[Account L1] * [Account].[Planning Account] * [Account].[Account] * [Account].[Stat Account] * [Account].[Stat Account Group]);

    ChannelAttribute : Select ( [Channel].[All Channel] * [Channel].[Stat Channel Group] * [Channel].[Stat Channel] * [Channel].[Channel L2] * [Channel].[Channel L1] * [Channel].[Planning Channel] * [Channel].[Channel]);

    RegionAttribute : Select ([Region].[All Region]*[Region].[Stat Region]*[Region].[Stat Region Group]*[Region].[Region L4]*[Region].[Region L3]*[Region].[Region L2]*[Region].[Region L1]*[Region].[Planning Region]*[Region].[Region]);

    PnLAttribute : Select ([PnL].[All PnL]*[PnL].[Stat PnL Group]*[PnL].[Stat PnL]*[PnL].[PnL L4]*[PnL].[PnL L3]*[PnL].[PnL L2]*[PnL].[PnL L1]*[PnL].[Planning PnL]*[PnL].[PnL]);

    DemandDomainAttribute : Select ([Demand Domain].[All Demand Domain]*[Demand Domain].[Stat Demand Domain Group]*[Demand Domain].[Stat Demand Domain]*[Demand Domain].[Transition Demand Domain]*[Demand Domain].[Demand Domain L4]*[Demand Domain].[Demand Domain L3]*[Demand Domain].[Demand Domain L2]*[Demand Domain].[Demand Domain L1]*[Demand Domain].[Planning Demand Domain]*[Demand Domain].[Demand Domain]);

    LocationAttribute : Select ([Location].[All Location]*[Location].[Stat Location Group]*[Location].[Stat Location]*[Location].[Location Type]*[Location].[Location Country]*[Location].[Location Region]*[Location].[Planning Location]*[Location].[Reporting Location]*[Location].[Location]);

Output Variables:
    LikeSkuResult

Slice Dimension Attributes:

"""

import logging

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

from o9_common_utils.O9DataLake import O9DataLake

logger = logging.getLogger("o9_logger")

import threading

from o9Reference.common_utils.o9_memory_utils import _get_memory

is_local = False

# Function Calls
Sales = O9DataLake.get("Sales")
ItemAttribute = O9DataLake.get("ItemAttribute")
PlanningItemAttribute = O9DataLake.get("PlanningItemAttribute")
Parameters = O9DataLake.get("Parameters")
AttributeWeights = O9DataLake.get("AttributeWeights")
AssortmentMapping = O9DataLake.get("AssortmentMapping")
ConsensusFcst = O9DataLake.get("ConsensusFcst")
GenerateSystemLikeItemMatchAssortment = O9DataLake.get("GenerateSystemLikeItemMatchAssortment")

# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

if not is_local:
    os.environ["FileStoreType"] = "google"
    import glob
    import logging
    import os
    import shutil
    from os import path

    from o9cloudutils import cloud_storage_utils, user_storage_path

    bucket = "bucket_slice {}".format(df_keys)
    local_storage_path = os.path.join(user_storage_path, bucket)
    test_folder_path = os.path.join(local_storage_path, "featureweights")
    os.makedirs(test_folder_path)

    try:
        """
        Code for storage pull
        """
        os.makedirs(test_folder_path, exist_ok=True)
        logger.debug("********List of files before storage_pull*******")
        logger.debug(os.listdir(local_storage_path))
        for filename in glob.iglob(local_storage_path + "**/**/*", recursive=True):
            logger.debug(filename)

        logger.info("Reading files from storage..............................................")
        local_storage_path = os.path.join(user_storage_path, bucket)
        value = cloud_storage_utils.storage_pull(bucket, local_storage_path, overwrite=True)

        logger.info("********List of files in bucket after storage pull************")
        for filename in glob.iglob(local_storage_path + "**/**/*", recursive=True):
            logger.debug(filename)

        input_df = pd.read_csv(os.path.join(test_folder_path, "featureweightsfull.csv"))
        logger.info(input_df.head())
    except:
        logger.exception("Couldn't pull folder")

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

from helpers.DP021SystemLikeAssortmentMatch import main

LikeSkuResult = main(
    sales=Sales,
    Item=ItemAttribute,
    pl_item=PlanningItemAttribute,
    Location=LocationAttribute,
    Account=AccountAttribute,
    Channel=ChannelAttribute,
    Region=RegionAttribute,
    DemandDomain=DemandDomainAttribute,
    PnL=PnLAttribute,
    parameters=Parameters,
    FullScopeWeights=input_df,
    FeatureWeights=AttributeWeights,
    SearchSpace=SearchSpace,
    AssortmentMapping=AssortmentMapping,
    numerical_cols=numerical_cols,
    FeatureLevel=FeatureLevel,
    ConsensusFcst=ConsensusFcst,
    ReadFromHive=ReadFromHive,
    generate_match_assortment=GenerateSystemLikeItemMatchAssortment,
    df_keys=df_keys,
)
O9DataLake.put("LikeSkuResult", LikeSkuResult)
