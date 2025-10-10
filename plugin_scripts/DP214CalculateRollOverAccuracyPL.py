"""
Plugin : DP214CalculateRollOverAccuracyPL
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    PlanningGrains- Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Location.[Planning Location]
    AccuracyWindow - 1
    InputTables - PlanningActual,SellOutActual
    DefaultMapping - Item-All Planning Item,Account-All Planning Account,Location-All Planning Location,Region-All Planning Region,Channel-All Planning Channel,Demand Domain-All Planning Demand Domain,PnL-All Planning PnL

Input Queries:
    PlanningActual : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row, ({Measure.[Actual]}) on column;

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    TimeDimension : select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastGenTimeBucket : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] ) on row,  ({Measure.[Forecast Generation Time Bucket]}) on column;

    ForecastIterationMasterData : Select (Version.[Version Name] * [Forecast Iteration].[Forecast Iteration Type] * [Forecast Iteration].[Forecast Iteration]) on row, ({Measure.[Iteration Type Input Stream]}) on column;

    StatBucketWeight : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration] * [Time].[Partial Week] ) on row,  ({Measure.[Stat Bucket Weight]}) on column;

    StatFcstPLWLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Week] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,
({Measure.[Stat Fcst PL W Lag]}) on column;

    StatFcstPLMLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Month] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,
({Measure.[Stat Fcst PL M Lag]}) on column;

    StatFcstPLPMLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Planning Month] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,
({Measure.[Stat Fcst PL PM Lag]}) on column;

    StatFcstPLQLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Quarter] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,
({Measure.[Stat Fcst PL Q Lag]}) on column;

    StatFcstPLPQLag : Select ([Version].[Version Name] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Time].[Planning Quarter] * [Location].[Planning Location] * [Account].[Planning Account] * [Demand Domain].[Planning Demand Domain] * [Lag].[Lag] * [PnL].[Planning PnL] * [Region].[Planning Region] * [Planning Cycle].[Planning Cycle Date] ) on row,
({Measure.[Stat Fcst PL PQ Lag]}) on column;

    SellOutActual : Select ([Version].[Version Name] * [Time].[Partial Week] * [Region].[Planning Region] * [Item].[Planning Item] * [Channel].[Planning Channel] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] ) on row,  ({Measure.[Sell Out Actual]}) on column;

Output Variables:
    StatFcstPLAbsError
    StatFcstPLLagAbsError
    StatFcstPLWLagAbsError
    StatFcstPLMLagAbsError
    StatFcstPLPMLagAbsError
    StatFcstPLQLagAbsError
    StatFcstPLPQLagAbsError

Slice Dimension Attributes: None

"""

import logging
import threading

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.dataframe_utils import convert_category_to_str
from o9Reference.common_utils.o9_memory_utils import _get_memory

from helpers.DP214CalculateRollOverAccuracyPL import main

logger = logging.getLogger("o9_logger")


# Function Calls
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
ForecastIterationMasterData = O9DataLake.get("ForecastIterationMasterData")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
StatBucketWeight = O9DataLake.get("StatBucketWeight")
TimeDimension = O9DataLake.get("TimeDimension")
StatFcstPLWLag = O9DataLake.get("StatFcstPLWLag")
StatFcstPLMLag = O9DataLake.get("StatFcstPLMLag")
StatFcstPLPMLag = O9DataLake.get("StatFcstPLPMLag")
StatFcstPLQLag = O9DataLake.get("StatFcstPLQLag")
StatFcstPLPQLag = O9DataLake.get("StatFcstPLPQLag")

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
        Input = convert_category_to_str(Input)
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
    StatFcstPLAbsError,
    StatFcstPLLagAbsError,
    StatFcstPLWLagAbsError,
    StatFcstPLMLagAbsError,
    StatFcstPLPMLagAbsError,
    StatFcstPLQLagAbsError,
    StatFcstPLPQLagAbsError,
) = main(
    PlanningActual=CombinedActual,
    StatFcstPLWLag=StatFcstPLWLag,
    StatFcstPLMLag=StatFcstPLMLag,
    StatFcstPLPMLag=StatFcstPLPMLag,
    StatFcstPLQLag=StatFcstPLQLag,
    StatFcstPLPQLag=StatFcstPLPQLag,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    ForecastIterationMasterData=ForecastIterationMasterData,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    PlanningGrains=PlanningGrains,
    StatBucketWeight=StatBucketWeight,
    AccuracyWindow=AccuracyWindow,
    df_keys=df_keys,
)

O9DataLake.put("StatFcstPLAbsError", StatFcstPLAbsError)
O9DataLake.put("StatFcstPLLagAbsError", StatFcstPLLagAbsError)
O9DataLake.put("StatFcstPLWLagAbsError", StatFcstPLWLagAbsError)
O9DataLake.put("StatFcstPLMLagAbsError", StatFcstPLMLagAbsError)
O9DataLake.put("StatFcstPLPMLagAbsError", StatFcstPLPMLagAbsError)
O9DataLake.put("StatFcstPLQLagAbsError", StatFcstPLQLagAbsError)
O9DataLake.put("StatFcstPLPQLagAbsError", StatFcstPLPQLagAbsError)
