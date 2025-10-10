"""
Plugin : DP065HorizonReconciliation
Version : 2025.04.00
Maintained by : dpref@o9solutions.com

Script Params:
    Grains - Item.[Planning Item],Account.[Planning Account],Channel.[Planning Channel],PnL.[Planning PnL],Region.[Planning Region],Demand Domain.[Planning Demand Domain],Location.[Planning Location]
    ReconcileLastBucket - False
    SellInOutputStream - Stat Fcst
    SellOutOutputStream - Sell Out Stat Fcst New
    OutputTables - SellIn_Output,SellOut_Output
    IterationTypesOrdered - Very Short Term,Short Term,Mid Term,Long Term,Very Long Term
    customRampUpWeight - None or any custom ramp up weight to be used for reconciliation

Input Queries:
    StatFcstPLAgg : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type] * [Item].[Planning Item] * [Account].[Planning Account] * [Channel].[Planning Channel] * [Region].[Planning Region] * [PnL].[Planning PnL] * [Location].[Planning Location] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] ) on row, ({Measure.[Stat Fcst PL Agg]}) on column;

    CMLFcstPLAgg : Select ([Forecast Iteration].[Forecast Iteration Type] * [Version].[Version Name] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Region].[Planning Region] * [Location].[Planning Location] * [Time].[Partial Week] * [Item].[Planning Item] ) on row,  ({Measure.[Stat Fcst PL CML Baseline Agg], Measure.[Stat Fcst PL CML External Driver Agg], Measure.[Stat Fcst PL CML Holiday Agg], Measure.[Stat Fcst PL CML Marketing Agg], Measure.[Stat Fcst PL CML Price Agg], Measure.[Stat Fcst PL CML Promo Agg], Measure.[Stat Fcst PL CML Residual Agg], Measure.[Stat Fcst PL CML Weather Agg]}) on column;

    HorizonEnd : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type] ) on row, ({Measure.[Forecast Generation Time Bucket], Measure.[Horizon End]}) on column;

    TimeDimension : Select ([Time].[Partial Week] * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    CurrentTimePeriod : select (&CurrentPartialWeek * [Time].[Week] * [Time].[Month] * [Time].[Planning Month] * [Time].[Quarter] * [Time].[Planning Quarter]) on row, () on column include memberproperties {Time.[Partial Week], Key} {Time.[Week], Key} {Time.[Month], Key} {Time.[Planning Month], Key} {Time.[Quarter], Key} {Time.[Planning Quarter], Key};

    ForecastIterationMasterData : Select ([Version].[Version Name] * [Forecast Iteration].[Forecast Iteration Type] * [Forecast Iteration].[Forecast Iteration]) on row,  ({Measure.[Iteration Type Input Stream],Measure.[Iteration Type Output Stream]}) on column;

    SellOut_Output:  Select ([Version].[Version Name] * [Region].[Planning Region] * [Channel].[Planning Channel] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] * [Account].[Planning Account] ) on row, ({Measure.[Sell Out Stat Fcst New]}) on column limit 1;

    SellIn_Output:  Select ([Version].[Version Name] * [Region].[Planning Region] * [Location].[Planning Location] * [Channel].[Planning Channel] * [PnL].[Planning PnL] * [Item].[Planning Item] * [Demand Domain].[Planning Demand Domain] * [Time].[Partial Week] * [Account].[Planning Account] ) on row,  ({Measure.[Stat Fcst]}) on column limit 1;

    MLDecompositionFlag : Select ([Forecast Iteration].[Forecast Iteration] * [Version].[Version Name] ) on row,  ({Measure.[CML Iteration Decomposition]}) on column;
    
    ReconciliationMethod: Select ([Data Stream].[Data Stream Type] * [Version].[Version Name] ) on row,  ({Measure.[Reconciliation Method]}) on column;
    
    ReconciliationTransitionPeriod: Select ([Forecast Iteration].[Forecast Iteration Type] * [Version].[Version Name]) on row,  ({Measure.[Reconciliation Transition Period]}) on column;

Output Variables:
    Output
    CMLOutput
    RampUpweightsOutput

Slice Dimension Attributes: None
"""

import logging

from o9Reference.common_utils.o9_memory_utils import _get_memory

logger = logging.getLogger("o9_logger")

import threading

from o9_common_utils.O9DataLake import O9DataLake

from helpers.DP065HorizonReconciliation import main

# Function Calls
StatFcstPLAgg = O9DataLake.get("StatFcstPLAgg")
CMLFcstPLAgg = O9DataLake.get("CMLFcstPLAgg")
HorizonEnd = O9DataLake.get("HorizonEnd")
TimeDimension = O9DataLake.get("TimeDimension")
CurrentTimePeriod = O9DataLake.get("CurrentTimePeriod")
ForecastIterationMasterData = O9DataLake.get("ForecastIterationMasterData")
SellOut_Output = O9DataLake.get("SellOut_Output")
SellIn_Output = O9DataLake.get("SellIn_Output")
MLDecompositionFlag = O9DataLake.get("MLDecompositionFlag")
ReconciliationMethod = O9DataLake.get("ReconciliationMethod")
ReconciliationTransitionPeriod = O9DataLake.get("ReconciliationTransitionPeriod")

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

Output, CMLOutput, RampUpweightsOutput = main(
    Grains=Grains,
    OutputTables=OutputTables,
    StatFcstPLAgg=StatFcstPLAgg,
    CMLFcstPLAgg=CMLFcstPLAgg,
    HorizonEnd=HorizonEnd,
    TimeDimension=TimeDimension,
    CurrentTimePeriod=CurrentTimePeriod,
    ReconcileLastBucket=ReconcileLastBucket,
    ForecastIterationMasterData=ForecastIterationMasterData,
    SellOut_Output=SellOut_Output,
    SellIn_Output=SellIn_Output,
    MLDecompositionFlag=MLDecompositionFlag,
    IterationTypesOrdered=IterationTypesOrdered,
    ReconciliationTransitionPeriod= ReconciliationTransitionPeriod,
    ReconciliationMethod=ReconciliationMethod,
    CustomRampUpWeight=CustomRampUpWeight,
    df_keys=df_keys,
)

O9DataLake.put("Output", Output)
O9DataLake.put("CMLOutput", CMLOutput)
O9DataLake.put("RampUpweightsOutput", RampUpweightsOutput)