"""
Plugin : DP045PopulateExchangeRate
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    None
-------------------
Input Queries:
    CurrencyExchangeRates - Select (Version.[Version Name]*FROM.[Currency].[Currency] * TO.[Currency].[Currency] * TO.[Time].[Month]) on row,
    ({Edge.[400 Exchange Rate Input].[Exchange Rate Input]}) on column where {RelationshipType.[400 Exchange Rate Input]};

    IsSingleCurrency - Select ([Version].[Version Name] ) on row,  ({Measure.[Is Single Currency]}) on column;

    DefaultInputCurrency - Select ([Version].[Version Name] ) on row,  ({Measure.[Default Input Currency]}) on column;

    CustomerGroupCurrency - select ([Region].[Planning Region]) on row, () on column include memberproperties {[Region].[Planning Region],[Local Currency]};

------------------------------------
Output Tables:
    CurrencyResult

PSEUDOCODE:
1. The program returns one output "CurrencyOutput" grained at two different levels based on Is Multi Currency
2. if Is Single Currency = 1 --> Customer Group is not needed in the grain (and CustomerGroupCurrency is not needed)
3. if Is Single Currency = 1 --> Only Customer Group is needed in the grain (and DefaultInputCurrency is not needed)
4. check if dataframe CustomerGroupCurrency or DefaultInputCurrency are empty
5. if Is Single Currency = 1:
6. create two dataframes
7. dataframe1 contains CurrencyExchangeRates where DefaultInputCurrency = from.Currency.Currency
with columns version, to currency,month,exchange rate
8. set exchange rate to 1
9. dataframe2 shopuld have values version, from currency, month, exchange rate
10. concat the two dataframes
11. If Is Single Currency = 0
12. take CurrencyExchangeRates (columns: version, to currency, motnh, 400 Exchange Rate input"
13. Take cross product with CustomerGroupCurrency
14. rename columns to relevant names and drop irrelevant columns (columns not needed in the grain
15. return CurrencyOutput

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP045PopulateExchangeRate import main

logger = logging.getLogger("o9_logger")


# Function Calls
CurrencyExchangeRates = O9DataLake.get("CurrencyExchangeRates")
IsSingleCurrency = O9DataLake.get("IsSingleCurrency")
DefaultInputCurrency = O9DataLake.get("DefaultInputCurrency")
CustomerGroupCurrency = O9DataLake.get("CustomerGroupCurrency")
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

CurrencyResult = main(
    CurrencyExchangeRates=CurrencyExchangeRates,
    IsSingleCurrency=IsSingleCurrency,
    DefaultInputCurrency=DefaultInputCurrency,
    CustomerGroupCurrency=CustomerGroupCurrency,
    df_keys=df_keys,
)

O9DataLake.put("CurrencyResult", CurrencyResult)
