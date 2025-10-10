"""
    Plugin : DP021SystemLikeItemMatch
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com

Script Params:

    FeatureLevel : Item.[L4]

    ReadFromHive : False

    numerical_cols : None

Input Queries:
    ItemAttribute :  Select ([Item].[Item]*[Item].[Planning Item]*[Item].[Transition Item]*[Item].[Stat Item]*[Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6]*[Item].[Item Stage]*[Item].[Item Type]*[Item].[PLC Status]*[Item].[Item Class]*[Item].[Transition Group]*[Item].[A1]*[Item].[A2]*[Item].[A3]*[Item].[A4]*[Item].[A5]*[Item].[A6]*[Item].[A7]*[Item].[A8]*[Item].[A9]*[Item].[A10]*[Item].[A11]*[Item].[A12]*[Item].[A13]*[Item].[A14]*[Item].[A15]*[Item].[All Item]) on row,() on column include memberproperties {[Item].[Item],[Item Intro Date]}{[Item].[Item],[Item Disc Date]}{[Item].[Item],[Item Status]}{[Item].[Item],[Is New Item]} INCLUDE_NULLMEMBERS;

    PlanningItemAttribute : Select ([Item].[Planning Item]) on row, () on column include memberproperties {[Item].[Planning Item],[Planning Item Intro Date]}{[Item].[Planning Item],[Planning Item Disc Date]}{[Item].[Planning Item],[Planning Item Locale]}{[Item].[Planning Item],[Base UOM]}{[Item].[Planning Item],[UOM Type]}{[Item].[Planning Item],[Container Size]}{[Item].[Planning Item],[Package Size]} INCLUDE_NULLMEMBERS;

    Parameters : Select ([Version].[Version Name]) on row, ({Measure.[Like Item Search History Measure], Measure.[Like Item Search Space], Measure.[Num Like Items], Measure.[Like Item History Period]}) on column;

    Sales : Select ([Version].[Version Name] * [Time].[Day] * [Region].[Planning Region] * [Item].[Item] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] ) on row, ({Measure.[Like Item Actual]}) on column;

    AttributeWeights : Select ([Item].[L4] * [Item Feature].[Item Feature] * [Version].[Version Name]) on row, ({Measure.[System Feature Weight], Measure.[User Feature Weight]}) on column;

    AssortmentMapping : Select ([Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location]* [Version].[Version Name]) on row, ({Measure.[Is Assorted]}) on column where {Measure.[Generate System Like Item Match Assortment]==1};

    ConsensusFcst : Select ([Version].[Version Name] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * &ConsensusForecastBuckets * [Item].[Planning Item] * [Location].[Planning Location] ) on row, ({Measure.[Consensus Fcst NPI BB]}) on column;

    GenerateSystemLikeItemMatchAssortment = Select ([Version].[Version Name] * [Region].[Planning Region] * [Location].[Planning Location] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row, ({Measure.[Generate System Like Item Match Assortment]}) on column;

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

from helpers.DP021SystemLikeItemMatch import main

LikeSkuResult = main(
    sales=Sales,
    Item=ItemAttribute,
    pl_item=PlanningItemAttribute,
    parameters=Parameters,
    FeatureWeights=AttributeWeights,
    AssortmentMapping=AssortmentMapping,
    numerical_cols=numerical_cols,
    FeatureLevel=FeatureLevel,
    ConsensusFcst=ConsensusFcst,
    ReadFromHive=ReadFromHive,
    generate_match_assortment=GenerateSystemLikeItemMatchAssortment,
    df_keys=df_keys,
)
O9DataLake.put("LikeSkuResult", LikeSkuResult)
