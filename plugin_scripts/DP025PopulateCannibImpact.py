"""
Plugin : DP025PopulateCannibImpact
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Input Queries:
//forecastData
Select (&AllPlanningItem * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * &CurrentAndFuturePartialWeeks * [Version].[Version Name].[CannibTesting] ) on row, ({Measure.[Stat Fcst NPI BB]}) on column;

//totalCombinations
Select (FROM.[Item].[Planning Item] * FROM.[Account].[Planning Account] * FROM.[Channel].[Planning Channel] * FROM.[Region].[Planning Region] * FROM.[PnL].[Planning PnL] *  FROM.[Location].[Planning Location] * TO.[Item].[Planning Item] * [Version].[Version Name].[CannibTesting]) on row, ({Edge.[610 Cannibalized Item Match].[Final Cannibalized Item Selected]}) on column where {RelationshipType.[610 Cannibalized Item Match], Edge.[610 Cannibalized Item Match].[Final Cannibalized Item Selected]==TRUE};

//NPIForecast
Select (&AllPlanningItem * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * &CurrentAndFuturePartialWeeks * [Version].[Version Name].[CannibTesting] ) on row, ({Measure.[NPI Fcst Final]}) on column;

//selectedNewItemCustomerCombination
Select ([Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Version].[Version Name].[CannibTesting] ) on row, ({Measure.[Populate Cannibalization Impact Assortment]}) on column;

//DefaultProfile
Select ([PLC Profile].[PLC Profile] * [Version].[Version Name].[CannibTesting] * [Lifecycle Time].[Lifecycle Bucket] ) on row, ({Measure.[Default Profile]}) on column include memberproperties {[PLC Profile].[PLC Profile], [PLC Time Bucket]} {[PLC Profile].[PLC Profile], [PLC Profile Type]} {[Lifecycle Time].[Lifecycle Bucket], Key};

//Parameters
Select ([Version].[Version Name].[CannibTesting] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Item].[Planning Item]) on row, ({Measure.[Intro Date],Measure.[Disco Date],Measure.[User Defined Cannibalization Period], Measure.[Cannibalization Profile], Measure.[Cannibalization Profile Bucket]}) on column;

//TimeDimension
Select ([Time].[Day] * [Time].[Week] * [Time].[Partial Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {[Time].[Week], Key} {[Time].[Partial Week], Key} {[Time].[Month], Key} {[Time].[Planning Month], Key};

Script Params:
//PlanningMonthFormat
Python date format of planning month column. Example: if planning month is M01-2018 then planning_month_format will be M%m-%Y

Output Variables:
cannibalizedForecast

Output Measures:
600 Cannibalization.[Cannib Impact]
600 Cannibalization.[Cannib Profile]
"""

import threading

import pandas as pd
from o9Reference.common_utils.o9_memory_utils import _get_memory

pd.options.display.max_rows = 25
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

import logging

logger = logging.getLogger("o9_logger")

from o9_common_utils.O9DataLake import O9DataLake

# Function Calls
ForecastData_df = O9DataLake.get("ForecastData")
TotalCombinations_df = O9DataLake.get("TotalCombinations")
NPIForecast_df = O9DataLake.get("NPIForecast")
SelectedNewItemCustomerCombination_df = O9DataLake.get("SelectedNewItemCustomerCombination")
DefaultProfile_df = O9DataLake.get("DefaultProfile")
Parameters_df = O9DataLake.get("Parameters")
TimeDimension_df = O9DataLake.get("TimeDimension")

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

from helpers.DP025PopulateCannibImpact import main

cannibalizedForecast, Cannibalization_Independence_Date = main(
    ForecastData=ForecastData_df,
    TotalCombinations=TotalCombinations_df,
    NPIForecast=NPIForecast_df,
    SelectedNewItemCustomerCombination=SelectedNewItemCustomerCombination_df,
    DefaultProfile=DefaultProfile_df,
    Parameters=Parameters_df,
    TimeDimension=TimeDimension_df,
    PlanningMonthFormat=PlanningMonthFormat,
    df_keys=df_keys,
)

O9DataLake.put("cannibalizedForecast", cannibalizedForecast)
O9DataLake.put("Cannibalization_Independence_Date", Cannibalization_Independence_Date)
