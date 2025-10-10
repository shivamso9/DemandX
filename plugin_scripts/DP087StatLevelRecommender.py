"""
Plugin : DP087StatLevelRecommender
Version : 2025.08.00
Maintained by : dpref@o9solutions.com

Script Params:
    VolSegLvl - A,B
    COVSegLvl - X,Y
    VolumeThresholdMeasure - Volume Threshold
    COVThresholdMeasure - COV Threshold
    SegmentationVolumeCalcGrain - Item.[Segmentation LOB]
    Grains - Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Location.[Stat Location],Item.[Stat Item]
    RequiredACFLagsInWeeks - 1,52
    ForecastScoreConfig - {'Spectral Ratio': [True, 0.1],  'ACF Mass': [True, 0.15],  'CoV': [True, 0.3],  'Intermittency': [True, 0.3], 'Signal to Noise Ratio': [True, 0.15]}
    Rule = {'Intermittency': False, 'CoV': False, 'Signal to Noise Ratio': True,'ACF Mass': True, 'Spectral Ratio': True}

Input Queries:

    ForecastParameters : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row, ({Measure.[Volume-COV History Period], Measure.[History Period], Measure.[COV Threshold], Measure.[Volume Threshold], Measure.[New Launch Period], Measure.[Disco Period]}) on column;

    ForecastGenTimeBucket : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row, ({Measure.[Forecast Generation Time Bucket]}) on column;

    StatLevels : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row, ({Measure.[Item Level], Measure.[Account Level], Measure.[Channel Level], Measure.[Location Level], Measure.[PnL Level], Measure.[Region Level], Measure.[Demand Domain Level]}) on column;

    StatActual : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] * [Region].[Stat Region] * [Demand Domain].[Stat Demand Domain] * [PnL].[Stat PnL] * [Channel].[Stat Channel] * [Account].[Stat Account] * [Location].[Stat Location] * [Time].[Partial Week] * [Item].[Stat Item] * [Item].[Segmentation LOB] ) on row, ({Measure.[Stat Actual]}) on column;

    TimeDimension : select ([Time].[Day] * [Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter] *[Time].[Week Name]) on row, () on column include memberproperties {[Time].[Day], Key} {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

Output Variables:
    ClassOutput
    ScoreOutput
    ProductSegmentL1

Slice Dimension Attributes:None
"""

import logging

from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

import threading

from o9_common_utils.O9DataLake import O9DataLake

from helpers.DP087StatLevelRecommender import main

# Function Calls
ForecastParameters = O9DataLake.get("ForecastParameters")
ForecastGenTimeBucket = O9DataLake.get("ForecastGenTimeBucket")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
TimeDimension = O9DataLake.get("TimeDimension")
StatActual = O9DataLake.get("StatActual")
StatLevels = O9DataLake.get("StatLevels")

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

ScoreOutput, ClassOutput, ProductSegmentL1 = main(
    Rule=Rule,
    Grains=Grains,
    VolSegLvl=VolSegLvl,
    COVSegLvl=COVSegLvl,
    VolumeThresholdMeasure=VolumeThresholdMeasure,
    COVThresholdMeasure=COVThresholdMeasure,
    SegmentationVolumeCalcGrain=SegmentationVolumeCalcGrain,
    RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
    ForecastScoreConfig=ForecastScoreConfig,
    ForecastParameters=ForecastParameters,
    ForecastGenTimeBucket=ForecastGenTimeBucket,
    TimeDimension=TimeDimension,
    StatActual=StatActual,
    StatLevels=StatLevels,
    CurrentTimePeriod=CurrentTimePeriod,
    df_keys=df_keys,
)

O9DataLake.put("ScoreOutput", ScoreOutput)
O9DataLake.put("ClassOutput", ClassOutput)
O9DataLake.put("ProductSegmentL1", ProductSegmentL1)
