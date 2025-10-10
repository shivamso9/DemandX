"""
Plugin : DP245DPHandshakeRateofSale_Pyspark.

Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:

Input Queries:
    HML : 514-1 HML Forecast Components
    SellingSeason : 045 Planning Item Customer Group

Output Variables:
    RateofSale - 514-1 HML Forecast Components
"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.pyspark.DP245DPHandshakeRateofSale_Pyspark import main

# Initialize logging
logger = logging.getLogger("o9_logger")

hml = O9DataLake.get("HML")
item_customer_group = O9DataLake.get("SellingSeason")

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.

back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

hml_rate_of_sale = main(
    hml=hml,
    item_customer_group=item_customer_group,
    spark=spark,
)

O9DataLake.put("RateofSale", hml_rate_of_sale)
