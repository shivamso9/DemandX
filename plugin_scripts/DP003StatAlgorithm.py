"""
    Plugin : DP003StatAlgorithm
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com

Script Params:
    MultiprocessingNumCores - 4

Input Queries:
    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    Input_Attribute_PlanningItem : Select ([Item].[Planning Item] * [Item].[Transition Item]*[Item].[Stat Item]* [Item].[L1]*[Item].[L2]*[Item].[L3]*[Item].[L4]*[Item].[L5]*[Item].[L6] * [Item].[All Item]) on row, () on column INCLUDE_NULLMEMBERS;

    Input_Attribute_Location : Select ([Location].[Location]*[Location].[Planning Location]*[Location].[Stat Location]*[Location].[Location Type]*[Location].[Location Country]*[Location].[All Location]) on row,() on column INCLUDE_NULLMEMBERS;

    Input_Cleansed_History : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] * [PnL].[Planning PnL] ) on row, ({Measure.[Stat Cleansed History]}) on column;

    Input_Stat_Level : Select ([Version].[Version Name] ) on row, ({Measure.[Stat Item Level], Measure.[Stat Location Level], Measure.[Stat Time Level], Measure.[Stat Channel Level], Measure.[Stat Region Level], Measure.[Stat Account Level], Measure.[Stat PnL Level], Measure.[Stat Demand Domain Level]}) on column;

    Input_Algorithm_Association : Select ([Stat Model].[Stat Model] * [Version].[Version Name] * [Stat Algorithm].[Stat Algorithm] ) on row,  ({Measure.[Stat Model Algorithm Association]}) on column;

    Input_Parameter_Value : Select ([Stat Model].[Stat Model] * [Stat Parameter].[Stat Parameter] * [Version].[Version Name] * [Stat Algorithm].[Stat Algorithm] ) on row,  ({Measure.[Stat Model Parameter Association], Measure.[Stat Parameter Value]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    Input_Attribute_Channel : Select(&AllPlanningChannel*[Channel].[Stat Channel]*[Channel].[Channel L1]*[Channel].[Channel L2]*[Channel].[All Channel]) on row, () on column INCLUDE_NULLMEMBERS ;

    Input_Attribute_Region : Select ([Region].[Planning Region]*[Region].[Stat Region]*[Region].[Region L1]*[Region].[Region L2]*[Region].[Region L3]*[Region].[Region L4]*[Region].[All Region]) on row, () on column INCLUDE_NULLMEMBERS;

    Input_Attribute_Account : Select( [Account].[Planning Account]*[Account].[Stat Account]* [Account].[Account L1]*[Account].[Account L2]*[Account].[Account L3]*[Account].[Account L4]*[Account].[All Account]) on row,() on column INCLUDE_NULLMEMBERS;

    Input_Attribute_PnL : select( [PnL].[Planning PnL]*[PnL].[Stat PnL]*[PnL].[PnL L1]*[PnL].[PnL L2]*[PnL].[PnL L3]*[PnL].[PnL L4]*[PnL].[All PnL]) on row, () on column INCLUDE_NULLMEMBERS;

    Input_Attribute_DemandDomain : select([Demand Domain].[Planning Demand Domain]*[Demand Domain].[Stat Demand Domain]*[Demand Domain].[Demand Domain L1]*[Demand Domain].[Demand Domain L2]*[Demand Domain].[Demand Domain L3]*[Demand Domain].[Demand Domain L4]*[Demand Domain].[All Demand Domain]) on row, () on column INCLUDE_NULLMEMBERS;

    DefaultAlgoParameters : Select ([Version].[Version Name] * [Stat Algorithm].[Stat Algorithm] * [Stat Parameter].[Stat Parameter] ) on row, ({Measure.[Stat Algorithm Parameter Association]}) on column include memberproperties {[Stat Parameter].[Stat Parameter], [Stat Parameter Weekly Default]} {[Stat Parameter].[Stat Parameter], [Stat Parameter Monthly Default]} {[Stat Parameter].[Stat Parameter], [Stat Parameter Quarterly Default]};

    MasterAlgoList : Select ([Version].[Version Name]) on row, ({Measure.[Assigned Algorithm List]}) on column;

Output Variables:
    OutputForecast
    OutputDescription

Slice Dimension Attributes:


"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP003StatAlgorithm import main

logger = logging.getLogger("o9_logger")

# Function Calls

logger.info("Reading data from o9DataLake ...")

CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
Input_Attribute_PlanningItem = O9DataLake.get("Input_Attribute_PlanningItem")
Input_Attribute_Location = O9DataLake.get("Input_Attribute_Location")
Input_Cleansed_History = O9DataLake.get("Input_Cleansed_History")
Input_Stat_Level = O9DataLake.get("Input_Stat_Level")
Input_Algorithm_Association = O9DataLake.get("Input_Algorithm_Association")
Input_Parameter_Value = O9DataLake.get("Input_Parameter_Value")
TimeDimension = O9DataLake.get("TimeDimension")
Input_Attribute_Channel = O9DataLake.get("Input_Attribute_Channel")
Input_Attribute_Region = O9DataLake.get("Input_Attribute_Region")
Input_Attribute_Account = O9DataLake.get("Input_Attribute_Account")
Input_Attribute_PnL = O9DataLake.get("Input_Attribute_PnL")
Input_Attribute_DemandDomain = O9DataLake.get("Input_Attribute_DemandDomain")
DefaultAlgoParameters = O9DataLake.get("DefaultAlgoParameters")

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
OutputForecast, OutputDescription = main(
    CurrentTimePeriod=CurrentTimePeriod,
    Input_Attribute_PlanningItem=Input_Attribute_PlanningItem,
    Input_Attribute_Location=Input_Attribute_Location,
    Input_Cleansed_History=Input_Cleansed_History,
    Input_Stat_Level=Input_Stat_Level,
    Input_Algorithm_Association=Input_Algorithm_Association,
    Input_Parameter_Value=Input_Parameter_Value,
    TimeDimension=TimeDimension,
    Input_Attribute_Channel=Input_Attribute_Channel,
    Input_Attribute_Region=Input_Attribute_Region,
    Input_Attribute_Account=Input_Attribute_Account,
    Input_Attribute_PnL=Input_Attribute_PnL,
    Input_Attribute_DemandDomain=Input_Attribute_DemandDomain,
    DefaultAlgoParameters=DefaultAlgoParameters,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
    df_keys=df_keys,
)

logger.info("Writing output to o9DataLake ...")
O9DataLake.put("OutputForecast", OutputForecast)
O9DataLake.put("OutputDescription", OutputDescription)
