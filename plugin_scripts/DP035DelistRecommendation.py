"""
Plugin : DP035DelistRecommendation
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params: None

Input Queries:
    PhaseOutHistoryPeriod : Select ([Version].[Version Name] ) on row,  ({Measure.[Phase Out History Period (in months)]}) on column;

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

    Actual_PlanningItem : Select ([Version].[Version Name] * &AllPastPartialWeeks * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location]* [Item].[L4] * [Item].[Planning Item] ) on row, ({Measure.[Actual]}) on column;

    Actual_L4: Select ([Version].[Version Name] * &AllPastPartialWeeks * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Location].[Planning Location] * [Item].[L4] ) on row, ({Measure.[Actual]}) on column;

Output Variables:
    RateOfChange

Slice Dimension Attributes: None

"""

import logging

logger = logging.getLogger("o9_logger")

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

from helpers.DP035DelistRecommendation import main

# Function Calls
RateOfChange = main(
    Actual_PlanningItem=Actual_PlanningItem,
    PhaseOutHistoryPeriod=PhaseOutHistoryPeriod,
    Actual_L4=Actual_L4,
    TimeDimension=TimeDimension,
    CurrentTimePeriod=CurrentTimePeriod,
)

O9DataLake.put("RateOfChange", RateOfChange)
