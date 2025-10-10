"""
Plugin : DP211CalculateBacktestPlanningLevelAccuracy
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    PlanningGrains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]
    BackTestCyclePeriod - 2,4,6
    InputTables - PlanningActual,SellOutActual
    DefaultMapping - Item-All Planning Item,Account-All Planning Account,Location-All Planning Location,Region-All Planning Region,Channel-All Planning Channel,Demand Domain-All Planning Demand Domain,PnL-All Planning PnL
    ReasonabilityCycles - 1

Input Queries:
    PlanningActual : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row, ({Measure.[Actual]}) on column;

    StatFcstPLLag1Backtest : Select ([Version].[Version Name] * [Time].[Partial Week] *[Region].[Planning Region] * [Item].[Planning Item] * [PnL].[Planning PnL] * [Forecast Iteration].[Forecast Iteration] * [Location].[Planning Location] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row, ({Measure.[Stat Fcst PL Lag1 Backtest]}) on column;

    StatFcstPLWLagBacktest : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Planning Region] * [Location].[Planning Location] * [Time].[Week] * [Lag].[Lag] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Planning Cycle].[Planning Cycle Date] * [Account].[Planning Account] ) on row,  ({Measure.[Stat Fcst PL W Lag Backtest]}) on column;

    StatFcstPLMLagBacktest : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Planning Region] * [Location].[Planning Location] * [Time].[Month] * [Lag].[Lag] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Planning Cycle].[Planning Cycle Date] * [Account].[Planning Account] ) on row,  ({Measure.[Stat Fcst PL M Lag Backtest]}) on column;

    StatFcstPLPMLagBacktest :Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Planning Region] * [Location].[Planning Location] * [Time].[Planning Month] * [Lag].[Lag] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Planning Cycle].[Planning Cycle Date] * [Account].[Planning Account] ) on row,  ({Measure.[Stat Fcst PL PM Lag Backtest]}) on column;

    StatFcstPLQLagBacktest : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Planning Region] * [Location].[Planning Location] * [Time].[Quarter] * [Lag].[Lag] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Planning Cycle].[Planning Cycle Date] * [Account].[Planning Account] ) on row,  ({Measure.[Stat Fcst PL Q Lag Backtest]}) on column;

    StatFcstPLPQLagBacktest : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Region].[Planning Region] * [Location].[Planning Location] * [Time].[Planning Quarter] * [Lag].[Lag] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Planning Cycle].[Planning Cycle Date] * [Account].[Planning Account] ) on row,  ({Measure.[Stat Fcst PL PQ Lag Backtest]}) on column;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    ForecastIterationMasterData : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Iteration Type Input Stream]}) on column;

    SellOutActual : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row,  ({Measure.[Sell Out Actual]}) on column include memberproperties {Time.[Partial Week],Key};

    PlanningCycleDates : Select ([Planning Cycle].[Planning Cycle Date]) on row, () on column include memberproperties{[Planning Cycle].[Planning Cycle Date],Key};

Output Variables:
    StatFcstPLLagBacktest
    StatFcstPLAbsErrorBacktest
    StatFcstPLLagAbsErrorBacktest
    StatFcstPLWLagAbsErrorBacktest
    StatFcstPLMLagAbsErrorBacktest
    StatFcstPLPMLagAbsErrorBacktest
    StatFcstPLQLagAbsErrorBacktest
    StatFcstPLPQLagAbsErrorBacktest
    StabilityOutput
    FcstNextNBucketsBacktest
    ActualsLastNBucketsBacktest

Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP211CalculateBacktestPlanningLevelAccuracy import main

logger = logging.getLogger("o9_logger")


# Function Calls
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
StatFcstPLLag1Backtest = O9DataLake.get("StatFcstPLLag1Backtest")
StatFcstPLWLagBacktest = O9DataLake.get("StatFcstPLWLagBacktest")
StatFcstPLMLagBacktest = O9DataLake.get("StatFcstPLMLagBacktest")
StatFcstPLPMLagBacktest = O9DataLake.get("StatFcstPLPMLagBacktest")
StatFcstPLQLagBacktest = O9DataLake.get("StatFcstPLQLagBacktest")
StatFcstPLPQLagBacktest = O9DataLake.get("StatFcstPLPQLagBacktest")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
TimeDimension = O9DataLake.get("TimeDimension")
ForecastIterationMasterData = O9DataLake.get("ForecastIterationMasterData")
PlanningCycleDates = O9DataLake.get("PlanningCycleDates")

DefaultMapping_list = DefaultMapping.strip().split(",")
default_mapping = {}
for mapping in DefaultMapping_list:
    key, value = mapping.split("-")
    default_mapping[key] = value

# configurable inputs
try:
    CombinedActual = {}
    AllInputTables = InputTables.strip().split(",")
    for i in AllInputTables:
        Input = O9DataLake.get(i)
        if Input is None:
            logger.exception(f"Input '{i}' is None, please recheck the table name...")
            continue
        CombinedActual[i] = Input
        logger.warning(f"Reading input {i}...")
except Exception as e:
    logger.exception(f"Error while reading inputs\nError:-\n{e}")

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

(
    StatFcstPLLagBacktest,
    StatFcstPLAbsErrorBacktest,
    StatFcstPLLagAbsErrorBacktest,
    StatFcstPLWLagAbsErrorBacktest,
    StatFcstPLMLagAbsErrorBacktest,
    StatFcstPLPMLagAbsErrorBacktest,
    StatFcstPLQLagAbsErrorBacktest,
    StatFcstPLPQLagAbsErrorBacktest,
    StabilityOutput,
    FcstNextNBucketsBacktest,
    ActualsLastNBucketsBacktest,
) = main(
    PlanningActual=CombinedActual,
    StatFcstPLLag1Backtest=StatFcstPLLag1Backtest,
    StatFcstPLWLagBacktest=StatFcstPLWLagBacktest,
    StatFcstPLMLagBacktest=StatFcstPLMLagBacktest,
    StatFcstPLPMLagBacktest=StatFcstPLPMLagBacktest,
    StatFcstPLQLagBacktest=StatFcstPLQLagBacktest,
    StatFcstPLPQLagBacktest=StatFcstPLPQLagBacktest,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    PlanningGrains=PlanningGrains,
    ReasonabilityCycles=ReasonabilityCycles,
    BackTestCyclePeriod=BackTestCyclePeriod,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    StatBucketWeight=StatBucketWeight,
    ForecastIterationMasterData=ForecastIterationMasterData,
    PlanningCycleDates=PlanningCycleDates,
    default_mapping=default_mapping,
    df_keys=df_keys,
)

O9DataLake.put("StatFcstPLLagBacktest", StatFcstPLLagBacktest)
O9DataLake.put("StatFcstPLAbsErrorBacktest", StatFcstPLAbsErrorBacktest)
O9DataLake.put("StatFcstPLLagAbsErrorBacktest", StatFcstPLLagAbsErrorBacktest)
O9DataLake.put("StatFcstPLWLagAbsErrorBacktest", StatFcstPLWLagAbsErrorBacktest)
O9DataLake.put("StatFcstPLMLagAbsErrorBacktest", StatFcstPLMLagAbsErrorBacktest)
O9DataLake.put("StatFcstPLPMLagAbsErrorBacktest", StatFcstPLPMLagAbsErrorBacktest)
O9DataLake.put("StatFcstPLQLagAbsErrorBacktest", StatFcstPLQLagAbsErrorBacktest)
O9DataLake.put("StatFcstPLPQLagAbsErrorBacktest", StatFcstPLPQLagAbsErrorBacktest)
O9DataLake.put("StabilityOutput", StabilityOutput)
O9DataLake.put("FcstNextNBucketsBacktest", FcstNextNBucketsBacktest)
O9DataLake.put("ActualsLastNBucketsBacktest", ActualsLastNBucketsBacktest)
