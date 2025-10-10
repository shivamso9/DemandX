"""
    Plugin : DP019MovingAverageSCPFcst
    Version : 0.0.0
    Maintained by : dpref@o9solutions.com

    Script Params:
        MovingAvgPeriods - 13
        FuturePeriods - 52
        HistoryMeasure - Sell In Stat L0
        Grains - Location.[Location],Item.[Item],Channel.[Planning Channel],Account.[Planning Account],PnL.[Planning PnL],Demand Domain.[Planning Demand Domain],Region.[Planning Region]
        TimeLevel - Time.[Week]
        OutputTimeLevel - Time.[Partial Week]
        MultiprocessingNumCores - 4

    Input Queries:
        History : Select ([Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain]* [Location].[Location] * [Version].[Version Name] * [Item].[Item] * [Time].[Week] ) on row,  ({Measure.[Sell In Stat L0]}) on column;

        CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

        TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key};

    Output Variables:
        output_df

"""

from logging import getLogger
from threading import Thread

from o9_common_utils.O9DataLake import O9DataLake
from o9Reference.common_utils.o9_memory_utils import _get_memory
from pandas import options, set_option

options.display.max_rows = 25
options.display.max_columns = 50
options.display.max_colwidth = 100
set_option("display.width", 1000)
options.display.precision = 3
options.mode.chained_assignment = None
logger = getLogger("o9_logger")


# Function Calls
logger.info("Reading data from o9DataLake ...")
History = O9DataLake.get("History")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}
else:
    logger.info("Slice : {}".format(df_keys))

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.
back_thread = Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

from helpers.DP019MovingAverageSCPFcst import main

output_df = main(
    MovingAvgPeriods=MovingAvgPeriods,
    FuturePeriods=FuturePeriods,
    HistoryMeasure=HistoryMeasure,
    Grains=Grains,
    TimeLevel=TimeLevel,
    all_history_df=History,
    CurrentTimePeriod=CurrentTimePeriod,
    TimeDimension=TimeDimension,
    df_keys=df_keys,
    OutputTimeLevel=OutputTimeLevel,
    multiprocessing_num_cores=int(MultiprocessingNumCores),
)
logger.info("Writing output to o9DataLake ...")
O9DataLake.put("output_df", output_df)
