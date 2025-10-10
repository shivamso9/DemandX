"""
Plugin : DP091DARespread
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]

Input Queries:
    InitiativeInput : Select (&CWVandDARespreadScenarios * [Initiative].[Initiative Type] * &AllDA) on row, ({Measure.[DA Item scope],Measure.[DA Alternate Item Level],Measure.[DA Alternate Item Scope],Measure.[DA Account Scope], Measure.[DA Channel Scope],Measure.[DA Region Scope],Measure.[DA PnL Scope], Measure.[DA Demand Domain Scope], Measure.[DA Location Scope],Measure.[DA Item Level],Measure.[DA Account Level], Measure.[DA Channel Level],Measure.[DA Region Level],Measure.[DA PnL Level], Measure.[DA Demand Domain Level], Measure.[DA Location Level], Measure.[DA Time Level],Measure.[DA End Date],Measure.[DA Start Date],Measure.[DA Assortment Basis],Measure.[DA Disaggregation Basis], Measure.[DA UOM]}) on column where{Measure.[DA End Date] > &CurrentDay.element(0).Key && (Measure.[Planner Status]== "In Forecast" || Measure.[Planner Status]== "In Forecast (via R&O)")};

    PlannerInput : Select (&CWVandDARespreadScenarios * &AllDA * [Time].[Partial Week] ) on row,  ({Measure.[Planner Input]}) on column;

    CandidateAssortment : Select (&CWVandDARespreadScenarios *[Item].[L6] * [Item].[L5] * [Item].[L4] * [Item].[L3]* [Item].[L2] * [Item].[L1] * [Item].[Planning Item] * [Item].[Item Type] * [Item].[All Item] * [Channel].[All Channel] * [Channel].[Planning Channel] *[Location].[All Location] * [Location].[Planning Location] * [Account].[All Account] * [Account].[Planning Account] * [Demand Domain].[All Demand Domain] * [Demand Domain].[Planning Demand Domain] * [PnL].[All PnL] * [PnL].[Planning PnL] * [Region].[All Region] * [Region].[Planning Region] ) on row,  ({Measure.[Assortment Final]}) on column;

    CandidateFcst : Select (&CWVandDARespreadScenarios *[Item].[L6] * [Item].[L5] * [Item].[L4] * [Item].[L3]* [Item].[L2] * [Item].[L1] * [Item].[Planning Item] * [Item].[Item Type] * [Item].[All Item] * [Channel].[All Channel] * [Channel].[Planning Channel] *[Time].[Partial Week] * [Location].[All Location] * [Location].[Planning Location] * [Account].[All Account] * [Account].[Planning Account] * [Demand Domain].[All Demand Domain] * [Demand Domain].[Planning Demand Domain] * [PnL].[All PnL] * [PnL].[Planning PnL] * [Region].[All Region] * [Region].[Planning Region] ) on row,  ({Measure.[Consensus Fcst],Measure.[Stat Fcst L0]}) on column;

    TimeDimension : select (Time.[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Planning Month] * [Time].[Month]) on row, () on column include memberproperties {Time.[Day],Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Planning Month],Key} {Time.[Month], Key};

    UOMConversion : Select (&CWVandDARespreadScenarios * [Item].[Planning Item] * [UOM].[UOM] ) on row, ({Measure.[UOM Conversion]}) on column;

    CurrentDay : Select (&CurrentDay) include memberproperties {Time.[Day],Key};

Output Variables:
    DAIntOutput

Slice Dimension Attributes:
    Version.[Version Name]

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP091DARespread import main

logger = logging.getLogger("o9_logger")

# Function Calls
InitiativeInput = O9DataLake.get("InitiativeInput")
PlannerInput = O9DataLake.get("PlannerInput")
CandidateAssortment = O9DataLake.get("CandidateAssortment")
CandidateFcst = O9DataLake.get("CandidateFcst")
TimeDimension = O9DataLake.get("TimeDimension")
UOMConversion = O9DataLake.get("UOMConversion")
CurrentDay = O9DataLake.get("CurrentDay")


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

DAIntOutput = main(
    InitiativeInput=InitiativeInput,
    PlannerInput=PlannerInput,
    CandidateAssortment=CandidateAssortment,
    CandidateFcst=CandidateFcst,
    TimeDimension=TimeDimension,
    Grains=Grains,
    CurrentDay=CurrentDay,
    UOMConversion=UOMConversion,
    df_keys=df_keys,
)

O9DataLake.put("DAIntOutput", DAIntOutput)
