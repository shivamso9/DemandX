import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data

from helpers.assign_forecast_rule import assign_rules
from helpers.DP009OutlierCleansing import processIteration as outlier_correction_main
from helpers.DP014GenerateValidationFcst import processIteration as validation_fcst_main
from helpers.DP015IdentifyBestFitModel import main as identify_best_fit
from helpers.DP015PopulateBestFitForecast import main as populate_best_fit_forecast
from helpers.DP015SystemStat import main as forecasting_main
from helpers.DP037SeasonalityDetection import (
    processIteration as seasonality_detection_main,
)
from helpers.DP056OutputChecks import processIteration as output_checks
from helpers.DP069EnsembleFcst import processIteration as calculate_ensemble_fcst
from helpers.o9Constants import o9Constants
from helpers.utils import (
    create_algo_list,
    filter_for_iteration,
    get_first_day_in_time_bucket,
)

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")


def check_for_new_intersections(
    the_last_time_period,
    the_idx,
    StatLevelActual,
    backtest_cycle_df,
    partial_week_key_col,
    forecast_level,
):
    # create a copy since we don't want to change the source df
    StatLevelActuals_copy = StatLevelActual.copy(deep=True)

    # get intersections in last cycle
    last_cycle_last_time_period = backtest_cycle_df[backtest_cycle_df.index == (the_idx - 1)][
        partial_week_key_col
    ].iloc[0]
    filter_clause = StatLevelActuals_copy[partial_week_key_col] <= last_cycle_last_time_period
    existing_intersections = StatLevelActuals_copy[filter_clause][forecast_level].drop_duplicates()

    logger.info(
        "Checking for new intersections introduced after {}...".format(last_cycle_last_time_period)
    )

    # get new intersections
    this_cycle_last_time_period = the_last_time_period[partial_week_key_col].iloc[0]
    filter_clause = StatLevelActuals_copy[partial_week_key_col] <= this_cycle_last_time_period
    new_intersections = StatLevelActuals_copy[filter_clause][forecast_level].drop_duplicates()

    # join and take only new intersections
    combined = existing_intersections.merge(new_intersections, how="right", indicator=True)

    # filter
    result = combined[combined["_merge"] == "right"]

    logger.info("Shape of new intersections : {}".format(result.shape))

    return result


col_mapping = {
    "Stat Fcst L1 Lag1 Backtest": float,
    "Stat Fcst L1 Lag Backtest": float,
    "Stat Fcst L1 LagN Backtest": float,
    "Actual Last N Buckets Backtest": float,
    "Fcst Next N Buckets Backtest": float,
    "Stat Fcst L1 Lag Backtest COCC": float,
    "Stat Fcst L1 Lag Backtest LC": float,
    "Stat Fcst L1 PM Lag Backtest": float,
    "Stat Fcst L1 M Lag Backtest": float,
    "Stat Fcst L1 W Lag Backtest": float,
    "SES W Lag Backtest": float,
    "DES W Lag Backtest": float,
    "TES W Lag Backtest": float,
    "ETS W Lag Backtest": float,
    "Auto ARIMA W Lag Backtest": float,
    "sARIMA W Lag Backtest": float,
    "Prophet W Lag Backtest": float,
    "STLF W Lag Backtest": float,
    "Theta W Lag Backtest": float,
    "Croston W Lag Backtest": float,
    "TBATS W Lag Backtest": float,
    "AR-NNET W Lag Backtest": float,
    "Simple Snaive W Lag Backtest": float,
    "Weighted Snaive W Lag Backtest": float,
    "Growth Snaive W Lag Backtest": float,
    "Naive Random Walk W Lag Backtest": float,
    "Seasonal Naive YoY W Lag Backtest": float,
    "Moving Average W Lag Backtest": float,
    "Simple AOA W Lag Backtest": float,
    "Growth AOA W Lag Backtest": float,
    "Weighted AOA W Lag Backtest": float,
    "SCHM W Lag Backtest": float,
    "SES M Lag Backtest": float,
    "DES M Lag Backtest": float,
    "TES M Lag Backtest": float,
    "ETS M Lag Backtest": float,
    "Auto ARIMA M Lag Backtest": float,
    "sARIMA M Lag Backtest": float,
    "Prophet M Lag Backtest": float,
    "STLF M Lag Backtest": float,
    "Theta M Lag Backtest": float,
    "Croston M Lag Backtest": float,
    "TBATS M Lag Backtest": float,
    "AR-NNET M Lag Backtest": float,
    "Simple Snaive M Lag Backtest": float,
    "Weighted Snaive M Lag Backtest": float,
    "Growth Snaive M Lag Backtest": float,
    "Naive Random Walk M Lag Backtest": float,
    "Seasonal Naive YoY M Lag Backtest": float,
    "Moving Average M Lag Backtest": float,
    "Simple AOA M Lag Backtest": float,
    "Growth AOA M Lag Backtest": float,
    "Weighted AOA M Lag Backtest": float,
    "SCHM M Lag Backtest": float,
    "SES PM Lag Backtest": float,
    "DES PM Lag Backtest": float,
    "TES PM Lag Backtest": float,
    "ETS PM Lag Backtest": float,
    "Auto ARIMA PM Lag Backtest": float,
    "sARIMA PM Lag Backtest": float,
    "Prophet PM Lag Backtest": float,
    "STLF PM Lag Backtest": float,
    "Theta PM Lag Backtest": float,
    "Croston PM Lag Backtest": float,
    "TBATS PM Lag Backtest": float,
    "AR-NNET PM Lag Backtest": float,
    "Simple Snaive PM Lag Backtest": float,
    "Weighted Snaive PM Lag Backtest": float,
    "Growth Snaive PM Lag Backtest": float,
    "Naive Random Walk PM Lag Backtest": float,
    "Seasonal Naive YoY PM Lag Backtest": float,
    "Moving Average PM Lag Backtest": float,
    "Simple AOA PM Lag Backtest": float,
    "Growth AOA PM Lag Backtest": float,
    "Weighted AOA PM Lag Backtest": float,
    "SCHM PM Lag Backtest": float,
}


def create_planning_cycles_from_vp_vf_vs(vp, vf, vs):
    try:
        vp_int = int(vp)
        vf_int = int(vf)
        vs_int = int(vs)
        cycles = [str(x) for x in range(vp_int, vp_int + vf_int * vs_int, vs_int)]
        return cycles
    except Exception as e:
        logger.warning(f"Error creating planning cycles from VP, VF, VS: {e}")
        return None


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    StatLevelActual,
    Parameters,
    SegmentationParameters,
    OutlierParameters,
    AlgoParameters,
    TimeDimension,
    CurrentTimePeriod,
    StatSegmentation,
    Rules,
    ForecastGenTimeBucket,
    StatBucketWeight,
    ReadFromHive,
    RUN_SEGMENTATION_EVERY_CYCLE,
    RUN_VALIDATION_EVERY_FOLD,
    RUN_BEST_FIT_EVERY_CYCLE,
    Grains,
    UseHolidays,
    IncludeDiscIntersections,
    OverrideFlatLineForecasts,
    MasterAlgoList,
    df_keys,
    alpha,
    DefaultAlgoParameters,
    BestFitSelectionCriteria,
    CurrentCycleBestfitAlgorithm,
    TrendVariationThreshold,
    LevelVariationThreshold,
    RangeVariationThreshold,
    SeasonalVariationThreshold,
    SeasonalVariationCountThreshold,
    MinimumIndicePercentage,
    AbsTolerance,
    COCCVariationThreshold,
    Weights,
    ReasonabilityCycles,
    ACFLowerThreshold,
    ACFUpperThreshold,
    ACFSkipLags,
    ACFDiff,
    RequiredACFLagsInWeeks,
    smooth_fraction,
    BackTestCyclePeriod,
    RolloverAlgorithmflag,
    LagsToStore,
    SystemEnsembleAlgorithmList,
    EnsembleParameters,
    IncludeEnsemble,
    EnsembleWeights,
    ForecastEngine,
    OutlierThresholds=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
    StatFcstCMLPlanningCycle=pd.DataFrame(),
    PlanningCycleDates=pd.DataFrame(),
    SeasonalIndices=None,
    multiprocessing_num_cores=4,
    ValidationParameters=pd.DataFrame(),
    CustomL1WLagBacktest=pd.DataFrame(),
    CustomL1MLagBacktest=pd.DataFrame(),
    CustomL1PMLagBacktest=pd.DataFrame(),
    CustomL1QLagBacktest=pd.DataFrame(),
    CustomL1PQLagBacktest=pd.DataFrame(),
    LagNInput="1",
    HistoryMeasure="Actual Cleansed",
    BacktestOnlyStatAlgos=True,
    BacktestOnlyCustomAlgos=False,
    the_iteration=None,
    TrendThreshold: int = 20,
):
    try:
        if o9Constants.FORECAST_ITERATION in ForecastGenTimeBucket.columns:
            Lag1FcstOutput_list = list()
            LagNFcstOutput_list = list()
            LagFcstOutput_list = list()
            AllForecastWithLagDim_list = list()
            SystemBestfitAlgorithmPlanningCycle_list = list()
            PlanningCycleAlgoStats_list = list()
            StabilityOutput_list = list()
            ReasonabilityOutput_list = list()

            for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
                logger.warning(f"--- Processing iteration {the_iteration}")

                decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

                (
                    the_lag_1_fcst_output,
                    the_lag_n_fcst_output,
                    the_lag_fcst_output,
                    the_all_forecast_with_lag_dim,
                    the_system_best_fit_pl_cycle,
                    the_planning_cycle_algo_stats,
                    the_stat_fcst_l1_lag_backtest_cocc,
                    the_reasonability_output,
                ) = decorated_func(
                    StatLevelActual=StatLevelActual,
                    Parameters=Parameters,
                    SegmentationParameters=SegmentationParameters,
                    OutlierParameters=OutlierParameters,
                    AlgoParameters=AlgoParameters,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    Rules=Rules,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    StatSegmentation=StatSegmentation,
                    StatBucketWeight=StatBucketWeight,
                    ReadFromHive=ReadFromHive,
                    RUN_SEGMENTATION_EVERY_CYCLE=RUN_SEGMENTATION_EVERY_CYCLE,
                    RUN_VALIDATION_EVERY_FOLD=RUN_VALIDATION_EVERY_FOLD,
                    RUN_BEST_FIT_EVERY_CYCLE=RUN_BEST_FIT_EVERY_CYCLE,
                    Grains=Grains,
                    UseHolidays=UseHolidays,
                    IncludeDiscIntersections=IncludeDiscIntersections,
                    OverrideFlatLineForecasts=OverrideFlatLineForecasts,
                    MasterAlgoList=MasterAlgoList,
                    df_keys=df_keys,
                    alpha=alpha,
                    DefaultAlgoParameters=DefaultAlgoParameters,
                    BestFitSelectionCriteria=BestFitSelectionCriteria,
                    CurrentCycleBestfitAlgorithm=CurrentCycleBestfitAlgorithm,
                    TrendVariationThreshold=TrendVariationThreshold,
                    LevelVariationThreshold=LevelVariationThreshold,
                    RangeVariationThreshold=RangeVariationThreshold,
                    SeasonalVariationThreshold=SeasonalVariationThreshold,
                    SeasonalVariationCountThreshold=SeasonalVariationCountThreshold,
                    MinimumIndicePercentage=MinimumIndicePercentage,
                    AbsTolerance=AbsTolerance,
                    COCCVariationThreshold=COCCVariationThreshold,
                    Weights=Weights,
                    ReasonabilityCycles=ReasonabilityCycles,
                    ACFLowerThreshold=ACFLowerThreshold,
                    ACFUpperThreshold=ACFUpperThreshold,
                    ACFSkipLags=ACFSkipLags,
                    ACFDiff=ACFDiff,
                    RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
                    smooth_fraction=smooth_fraction,
                    BackTestCyclePeriod=BackTestCyclePeriod,
                    RolloverAlgorithmflag=RolloverAlgorithmflag,
                    LagsToStore=LagsToStore,
                    multiprocessing_num_cores=multiprocessing_num_cores,
                    StatFcstCMLPlanningCycle=StatFcstCMLPlanningCycle,
                    OutlierThresholds=OutlierThresholds,
                    PlanningCycleDates=PlanningCycleDates,
                    CustomL1WLagBacktest=CustomL1WLagBacktest,
                    ValidationParameters=ValidationParameters,
                    CustomL1MLagBacktest=CustomL1MLagBacktest,
                    CustomL1PMLagBacktest=CustomL1PMLagBacktest,
                    CustomL1QLagBacktest=CustomL1QLagBacktest,
                    CustomL1PQLagBacktest=CustomL1PQLagBacktest,
                    SystemEnsembleAlgorithmList=SystemEnsembleAlgorithmList,
                    EnsembleParameters=EnsembleParameters,
                    IncludeEnsemble=IncludeEnsemble,
                    EnsembleWeights=EnsembleWeights,
                    ForecastEngine=ForecastEngine,
                    SeasonalIndices=SeasonalIndices,
                    SellOutOffset=SellOutOffset,
                    LagNInput=LagNInput,
                    HistoryMeasure=HistoryMeasure,
                    BacktestOnlyStatAlgos=BacktestOnlyStatAlgos,
                    BacktestOnlyCustomAlgos=BacktestOnlyCustomAlgos,
                    the_iteration=the_iteration,
                    TrendThreshold=TrendThreshold,
                )
                Lag1FcstOutput_list.append(the_lag_1_fcst_output)
                LagNFcstOutput_list.append(the_lag_n_fcst_output)
                LagFcstOutput_list.append(the_lag_fcst_output)
                AllForecastWithLagDim_list.append(the_all_forecast_with_lag_dim)
                SystemBestfitAlgorithmPlanningCycle_list.append(the_system_best_fit_pl_cycle)
                PlanningCycleAlgoStats_list.append(the_planning_cycle_algo_stats)
                StabilityOutput_list.append(the_stat_fcst_l1_lag_backtest_cocc)
                ReasonabilityOutput_list.append(the_reasonability_output)

            Lag1FcstOutput = concat_to_dataframe(Lag1FcstOutput_list)
            LagNFcstOutput = concat_to_dataframe(LagNFcstOutput_list)
            LagFcstOutput = concat_to_dataframe(LagFcstOutput_list)
            AllForecastWithLagDim = concat_to_dataframe(AllForecastWithLagDim_list)
            SystemBestfitAlgorithmPlanningCycle = concat_to_dataframe(
                SystemBestfitAlgorithmPlanningCycle_list
            )
            PlanningCycleAlgoStats = concat_to_dataframe(PlanningCycleAlgoStats_list)
            StabilityOutput = concat_to_dataframe(StabilityOutput_list)
            ReasonabilityOutput = concat_to_dataframe(ReasonabilityOutput_list)

        else:
            (
                Lag1FcstOutput,
                LagNFcstOutput,
                LagFcstOutput,
                AllForecastWithLagDim,
                SystemBestfitAlgorithmPlanningCycle,
                PlanningCycleAlgoStats,
                StabilityOutput,
                ReasonabilityOutput,
            ) = processIteration(
                StatLevelActual=StatLevelActual,
                Parameters=Parameters,
                SegmentationParameters=SegmentationParameters,
                OutlierParameters=OutlierParameters,
                AlgoParameters=AlgoParameters,
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                Rules=Rules,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                StatBucketWeight=StatBucketWeight,
                StatSegmentation=StatSegmentation,
                ReadFromHive=ReadFromHive,
                RUN_SEGMENTATION_EVERY_CYCLE=RUN_SEGMENTATION_EVERY_CYCLE,
                RUN_VALIDATION_EVERY_FOLD=RUN_VALIDATION_EVERY_FOLD,
                RUN_BEST_FIT_EVERY_CYCLE=RUN_BEST_FIT_EVERY_CYCLE,
                Grains=Grains,
                UseHolidays=UseHolidays,
                IncludeDiscIntersections=IncludeDiscIntersections,
                OverrideFlatLineForecasts=OverrideFlatLineForecasts,
                MasterAlgoList=MasterAlgoList,
                df_keys=df_keys,
                alpha=alpha,
                DefaultAlgoParameters=DefaultAlgoParameters,
                BestFitSelectionCriteria=BestFitSelectionCriteria,
                CurrentCycleBestfitAlgorithm=CurrentCycleBestfitAlgorithm,
                TrendVariationThreshold=TrendVariationThreshold,
                LevelVariationThreshold=LevelVariationThreshold,
                RangeVariationThreshold=RangeVariationThreshold,
                SeasonalVariationThreshold=SeasonalVariationThreshold,
                SeasonalVariationCountThreshold=SeasonalVariationCountThreshold,
                MinimumIndicePercentage=MinimumIndicePercentage,
                AbsTolerance=AbsTolerance,
                COCCVariationThreshold=COCCVariationThreshold,
                Weights=Weights,
                ReasonabilityCycles=ReasonabilityCycles,
                ACFLowerThreshold=ACFLowerThreshold,
                ACFUpperThreshold=ACFUpperThreshold,
                ACFSkipLags=ACFSkipLags,
                ACFDiff=ACFDiff,
                RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
                smooth_fraction=smooth_fraction,
                BackTestCyclePeriod=BackTestCyclePeriod,
                RolloverAlgorithmflag=RolloverAlgorithmflag,
                LagsToStore=LagsToStore,
                multiprocessing_num_cores=multiprocessing_num_cores,
                StatFcstCMLPlanningCycle=StatFcstCMLPlanningCycle,
                OutlierThresholds=OutlierThresholds,
                PlanningCycleDates=PlanningCycleDates,
                ValidationParameters=ValidationParameters,
                CustomL1WLagBacktest=CustomL1WLagBacktest,
                CustomL1MLagBacktest=CustomL1MLagBacktest,
                CustomL1PMLagBacktest=CustomL1PMLagBacktest,
                CustomL1QLagBacktest=CustomL1QLagBacktest,
                CustomL1PQLagBacktest=CustomL1PQLagBacktest,
                SystemEnsembleAlgorithmList=SystemEnsembleAlgorithmList,
                EnsembleParameters=EnsembleParameters,
                IncludeEnsemble=IncludeEnsemble,
                EnsembleWeights=EnsembleWeights,
                ForecastEngine=ForecastEngine,
                SeasonalIndices=SeasonalIndices,
                SellOutOffset=SellOutOffset,
                LagNInput=LagNInput,
                HistoryMeasure=HistoryMeasure,
                BacktestOnlyStatAlgos=BacktestOnlyStatAlgos,
                BacktestOnlyCustomAlgos=BacktestOnlyCustomAlgos,
                TrendThreshold=TrendThreshold,
                the_iteration=None,
            )

    except Exception as e:
        logger.exception(e)
        Lag1FcstOutput, LagNFcstOutput, LagFcstOutput = None, None, None
        (
            AllForecastWithLagDim,
            SystemBestfitAlgorithmPlanningCycle,
            PlanningCycleAlgoStats,
            StabilityOutput,
            ReasonabilityOutput,
        ) = (
            None,
            None,
            None,
            None,
            None,
        )

    return (
        Lag1FcstOutput,
        LagNFcstOutput,
        LagFcstOutput,
        AllForecastWithLagDim,
        SystemBestfitAlgorithmPlanningCycle,
        PlanningCycleAlgoStats,
        StabilityOutput,
        ReasonabilityOutput,
    )


def processIteration(
    StatLevelActual,
    Parameters,
    SegmentationParameters,
    OutlierParameters,
    AlgoParameters,
    TimeDimension,
    CurrentTimePeriod,
    StatSegmentation,
    Rules,
    ForecastGenTimeBucket,
    StatBucketWeight,
    ReadFromHive,
    RUN_SEGMENTATION_EVERY_CYCLE,
    RUN_VALIDATION_EVERY_FOLD,
    RUN_BEST_FIT_EVERY_CYCLE,
    Grains,
    UseHolidays,
    IncludeDiscIntersections,
    OverrideFlatLineForecasts,
    MasterAlgoList,
    df_keys,
    alpha,
    DefaultAlgoParameters,
    BestFitSelectionCriteria,
    CurrentCycleBestfitAlgorithm,
    TrendVariationThreshold,
    LevelVariationThreshold,
    RangeVariationThreshold,
    SeasonalVariationThreshold,
    SeasonalVariationCountThreshold,
    MinimumIndicePercentage,
    AbsTolerance,
    COCCVariationThreshold,
    Weights,
    ReasonabilityCycles,
    ACFLowerThreshold,
    ACFUpperThreshold,
    ACFSkipLags,
    ACFDiff,
    RequiredACFLagsInWeeks,
    smooth_fraction,
    BackTestCyclePeriod,
    RolloverAlgorithmflag,
    LagsToStore,
    SystemEnsembleAlgorithmList,
    EnsembleParameters,
    IncludeEnsemble,
    EnsembleWeights,
    ForecastEngine,
    SeasonalIndices=None,
    StatFcstCMLPlanningCycle=pd.DataFrame(),
    OutlierThresholds=pd.DataFrame(),
    PlanningCycleDates=pd.DataFrame(),
    ValidationParameters=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
    multiprocessing_num_cores=4,
    CustomL1WLagBacktest=pd.DataFrame(),
    CustomL1MLagBacktest=pd.DataFrame(),
    CustomL1PMLagBacktest=pd.DataFrame(),
    CustomL1QLagBacktest=pd.DataFrame(),
    CustomL1PQLagBacktest=pd.DataFrame(),
    LagNInput="1",
    HistoryMeasure="Actual Cleansed",
    BacktestOnlyStatAlgos=True,
    BacktestOnlyCustomAlgos=False,
    the_iteration=None,
    TrendThreshold: int = 20,
):
    plugin_name = "DP027BacktestL1"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    lag_col = "Lag.[Lag]"
    stat_parameter_col = "Stat Parameter.[Stat Parameter]"
    stat_algorithm_col = "Stat Algorithm.[Stat Algorithm]"
    system_stat_param_value_col = "System Stat Parameter Value"
    outlier_correction_col = "Outlier Correction"
    outlier_upper_threshold_limit_col = "Outlier Upper Threshold Limit"
    outlier_lower_threshold_limit_col = "Outlier Lower Threshold Limit"
    outlier_method_col = "Outlier Method"
    best_fit_method_col = "Bestfit Method"
    disco_period_col = "Disco Period"
    error_metric_col = "Error Metric"
    forecast_period_col = "Forecast Period"
    history_measure_col = "History Measure"
    history_period_col = "History Period"
    history_time_bucket_col = "History Time Buckets"
    intermittency_threshold_col = "Intermittency Threshold"
    new_launch_period_col = "New Launch Period"
    seasonality_threshold_col = "Seasonality Threshold"
    trend_threshold_col = "Trend Threshold"
    validation_period_col = "Validation Period"
    vol_cov_history_period_col = "Volume-COV History Period"
    actual_cleansed_system_col = "Actual Cleansed System"
    class_col = "Class.[Class]"
    vol_threshold_col = "Volume Threshold"
    cov_threshold_col = "COV Threshold"
    forecast_period_col = "Forecast Period"
    system_best_fit_algo_col = "System Bestfit Algorithm"
    planner_best_fit_algo_col = "Planner Bestfit Algorithm"
    seasonality_l1_col = "Seasonality L1"
    prod_cust_segment_l1_col = "Product Customer L1 Segment"
    system_algo_list_col = "Stat Rule.[System Algorithm List]"
    planner_algo_list_col = "Planner Algorithm List"
    intermittent_l1_col = "Intermittent L1"
    plc_status_l1_col = "PLC Status L1"
    vol_segment_l1_col = "Volume Segment L1"
    cov_segment_l1_col = "COV Segment L1"
    trend_l1_col = "Trend L1"
    seasonality_l1_col = "Seasonality L1"
    stat_rule_col = "Stat Rule.[Stat Rule]"
    planner_vol_segment_col = "Planner Volume Segment"
    planner_cov_segment_col = "Planner COV Segment"
    planner_intermittency_col = "Planner Intermittency"
    planner_plc_col = "Planner PLC"
    planner_trend_col = "Planner Trend"
    planner_seasonality_col = "Planner Seasonality"
    planner_algo_list_col = "Planner Algorithm List"
    assigned_algo_list_col = "Assigned Algorithm List"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    partial_week_col = "Time.[Partial Week]"
    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"
    quarter_col = "Time.[Quarter]"
    planning_quarter_col = "Time.[Planning Quarter]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"
    los_l1_col = "Length of Series L1"
    planner_los_col = "Planner Length of Series"
    system_stat_fcst_l1_col = "System Stat Fcst L1"
    actual_cleansed_col = "Actual Cleansed"
    PLANNER_STAT_FORECAST: str = "Planner Stat Fcst L1"
    STAT_FCST_L1_LC: str = "Stat Fcst L1 LC"
    los_l1_integer_col = "Length of Series L1 Integer"
    segmentation_lob_col = "Item.[Segmentation LOB]"
    stat_item_col = "Item.[Stat Item]"

    validation_actual_col = "Validation Actual"
    validation_fcst_col = "Validation Fcst"
    composite_error_col = "Composite Error"
    validation_fcst_abs_error_col = "Validation Fcst Abs Error"
    validation_error_col = "Validation Error"
    actual_last_n_buckets = "Actual Last N Buckets"
    fcst_next_n_buckets = "Fcst Next N Buckets"
    actual_last_n_buckets_backtest = "Actual Last N Buckets Backtest"
    fcst_next_n_buckets_backtest = "Fcst Next N Buckets Backtest"
    bestfit_selection_criteria_col = "Bestfit Selection Criteria"
    system_ensemble_algo_list_col = "System Ensemble Algorithm List"
    forecast_strategy_col = "Forecast Strategy"
    sell_out_offset_col = "Offset Period"
    cocc_measure = "Stat Fcst L1 Lag Backtest COCC"
    lc_measure = "Stat Fcst L1 Lag Backtest LC"
    stat_fcst_l1_lag_col_PW = "Stat Fcst L1 Lag Backtest"

    # SCHM
    seasonal_index = "SCHM Seasonal Index"
    validation_seasonal_index = "SCHM Validation Seasonal Index"

    seasonal_index_backtest = "SCHM Seasonal Index Backtest"
    validation_seasonal_index_backtest = "SCHM Validation Seasonal Index Backtest"

    # CML
    stat_fcst_prefix = "Stat Fcst "
    lower_bound_suffix = " 80% LB"
    upper_bound_suffix = " 80% UB"

    if BacktestOnlyStatAlgos and BacktestOnlyCustomAlgos:
        logger.warning("BacktestOnlyStatAlgos and BacktestOnlyStatAlgos both can't be true ...")
        logger.warning("Considering only BacktestOnlyStatAlgos as True ...")
        BacktestOnlyCustomAlgos = False

    # output measures
    stat_fcst_l1_lag1_col = "Stat Fcst L1 Lag1 Backtest"

    # add all algos to MasterAlgoList to avoid discrepancy with planner algo list/different slices
    MasterAlgoList = pd.DataFrame(
        {
            assigned_algo_list_col: "Seasonal Naive YoY,AR-NNET,Moving Average,Naive Random Walk,STLF,Auto ARIMA,sARIMA,Prophet,SES,DES,TES,ETS,Croston,Theta,TBATS,Simple Snaive,Weighted Snaive,Growth Snaive,Simple AOA,Weighted AOA,Growth AOA,SCHM"
        },
        index=[0],
    )

    # Add names of new algorithms here
    ALGO_MASTER_LIST = create_algo_list(AlgoDF=MasterAlgoList, algo_list_col=assigned_algo_list_col)

    ALL_STAT_FORECAST_LB_COLUMNS = ["Stat Fcst " + x + " 80% LB" for x in ALGO_MASTER_LIST]
    ALL_STAT_FORECAST_UB_COLUMNS = ["Stat Fcst " + x + " 80% UB" for x in ALGO_MASTER_LIST]

    ALL_STAT_FCST_BOUND_COLUMNS = ALL_STAT_FORECAST_LB_COLUMNS + ALL_STAT_FORECAST_UB_COLUMNS
    ALL_STAT_FORECAST_COLUMNS = ["Stat Fcst " + x for x in ALGO_MASTER_LIST]

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]

    logger.info("forecast_level : {}".format(forecast_level))

    # filter required columns
    req_cols_from_outlier_output = (
        [version_col] + forecast_level + [partial_week_col, actual_cleansed_system_col]
    )
    non_disc_intersections_req_cols = (
        [version_col, class_col] + forecast_level + [prod_cust_segment_l1_col]
    )
    # Collect actuals and forecasts in the same dataframe as required by best fit
    join_cols = [version_col] + forecast_level + [partial_week_col]

    # Collect forecast bound columns
    ForecastBounds_req_cols = (
        [version_col] + forecast_level + [partial_week_col] + ALL_STAT_FCST_BOUND_COLUMNS
    )
    # join back with stat segmentation
    req_seasonality_cols = forecast_level + [seasonality_l1_col, trend_l1_col]

    # infer time related attributes from forecast gen time bucket
    fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]
    logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

    # Default is planning month
    relevant_time_cols = [
        partial_week_col,
        partial_week_key_col,
        planning_month_col,
        planning_month_key_col,
    ]
    relevant_time_name = planning_month_col
    relevant_time_key = planning_month_key_col
    lag_suffix = " PM Lag Backtest"

    if fcst_gen_time_bucket == "Week":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            week_col,
            week_key_col,
        ]
        relevant_time_name = week_col
        relevant_time_key = week_key_col
        lag_suffix = " W Lag Backtest"
    elif fcst_gen_time_bucket == "Month":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            month_col,
            month_key_col,
        ]
        relevant_time_name = month_col
        relevant_time_key = month_key_col
        lag_suffix = " M Lag Backtest"
    elif fcst_gen_time_bucket == "Quarter":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            quarter_col,
            quarter_key_col,
        ]
        relevant_time_name = quarter_col
        relevant_time_key = quarter_key_col
        lag_suffix = " Q Lag Backtest"
    elif fcst_gen_time_bucket == "Planning Quarter":
        relevant_time_cols = [
            partial_week_col,
            partial_week_key_col,
            planning_quarter_col,
            planning_quarter_key_col,
        ]
        relevant_time_name = planning_quarter_col
        relevant_time_key = planning_quarter_key_col
        lag_suffix = " PQ Lag Backtest"

    # define lag measures which are dependant on time grain
    ALL_STAT_FCST_LAG_COLUMNS = [x + lag_suffix for x in ALGO_MASTER_LIST]
    stat_fcst_l1_lag_col = "Stat Fcst L1" + lag_suffix

    # Convert Length of Series value into categorical value
    los_for_2_cycles = 24
    if "Week" in fcst_gen_time_bucket:
        los_for_2_cycles = 104
    elif "Quarter" in fcst_gen_time_bucket:
        los_for_2_cycles = 8

    Lag1FcstOutput_cols = [version_col] + forecast_level + [partial_week_col, stat_fcst_l1_lag1_col]
    Lag1FcstOutput = pd.DataFrame(columns=Lag1FcstOutput_cols)

    LagNFcstOutput_cols = (
        [version_col]
        + forecast_level
        + [
            partial_week_col,
            o9Constants.STAT_FCST_L1_LAGN_BACKTEST,
        ]
    )
    LagNFcstOutput = pd.DataFrame(columns=LagNFcstOutput_cols)
    LagFcstOutput_cols = (
        [version_col]
        + forecast_level
        + [partial_week_col, o9Constants.PLANNING_CYCLE_DATE, stat_fcst_l1_lag_col_PW]
    )
    LagFcstOutput = pd.DataFrame(columns=LagFcstOutput_cols)

    AllForecastWithLagDim_cols = (
        [version_col]
        + forecast_level
        + [
            o9Constants.PLANNING_CYCLE_DATE,
            lag_col,
            relevant_time_name,
            stat_fcst_l1_lag_col,
        ]
        + ALL_STAT_FCST_LAG_COLUMNS
    )
    AllForecastWithLagDim = pd.DataFrame(columns=AllForecastWithLagDim_cols)

    SystemBestfitAlgorithmPlanningCycle_cols = (
        [version_col]
        + forecast_level
        + [
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.SYSTEM_BESTFIT_ALGORITHM_PLANNING_CYCLE,
        ]
    )
    SystemBestfitAlgorithmPlanningCycle = pd.DataFrame(
        columns=SystemBestfitAlgorithmPlanningCycle_cols
    )

    PlanningCycleAlgoStats_cols = (
        [version_col]
        + forecast_level
        + [
            o9Constants.PLANNING_CYCLE_DATE,
            o9Constants.STAT_ALGORITHM,
            o9Constants.COMPOSITE_ERROR_PLANNING_CYCLE,
            o9Constants.VALIDATION_ERROR_PLANNING_CYCLE,
            o9Constants.ALGORITHM_PARAMETERS_PLANNING_CYCLE,
            o9Constants.VALIDATION_METHOD_PLANNING_CYCLE,
        ]
    )
    PlanningCycleAlgoStats = pd.DataFrame(columns=PlanningCycleAlgoStats_cols)
    StabilityOutput_cols = (
        [version_col]
        + forecast_level
        + [o9Constants.PLANNING_CYCLE_DATE, partial_week_col, cocc_measure, lc_measure]
    )
    StabilityOutput = pd.DataFrame(columns=StabilityOutput_cols)
    ReasonabilityOutput_cols = (
        [version_col]
        + forecast_level
        + [
            o9Constants.PLANNING_CYCLE_DATE,
            actual_last_n_buckets_backtest,
            fcst_next_n_buckets_backtest,
        ]
    )
    ReasonabilityOutput = pd.DataFrame(columns=ReasonabilityOutput_cols)
    try:
        BackTestCyclePeriods = [int(x.strip()) for x in BackTestCyclePeriod.split(",")]

        # getting forecat engine to check whether to run proc for cycle or not
        forecast_engine = ForecastEngine[o9Constants.FORECAST_ENGINE].iloc[0]
        run_for_cycle = True
        if forecast_engine == "ML":
            run_for_cycle = False

        if not BackTestCyclePeriods:
            logger.warning("BackTestCyclePeriods not populated")
            return (
                Lag1FcstOutput,
                LagNFcstOutput,
                AllForecastWithLagDim,
                SystemBestfitAlgorithmPlanningCycle,
                PlanningCycleAlgoStats,
                StabilityOutput,
                ReasonabilityOutput,
            )
        if len(SellOutOffset) == 0:
            logger.warning(
                f"Empty SellOut offset input for the forecast iteration {the_iteration}, assuming offset as 0 ..."
            )
            SellOutOffset = pd.DataFrame(
                {
                    o9Constants.VERSION_NAME: [
                        ForecastGenTimeBucket[o9Constants.VERSION_NAME].values[0]
                    ],
                    sell_out_offset_col: [0],
                }
            )

        # assert and convert string value to boolean
        assert ReadFromHive in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(ReadFromHive)

        seg_to_planner_col_mapping = {
            intermittent_l1_col: planner_intermittency_col,
            plc_status_l1_col: planner_plc_col,
            los_l1_col: planner_los_col,
            cov_segment_l1_col: planner_cov_segment_col,
            vol_segment_l1_col: planner_vol_segment_col,
            trend_l1_col: planner_trend_col,
            seasonality_l1_col: planner_seasonality_col,
        }

        # Filter only relevant columns
        req_cols = (
            [version_col, stat_parameter_col]
            + forecast_level
            + [stat_algorithm_col, system_stat_param_value_col]
        )
        AlgoParameters = AlgoParameters[req_cols]

        req_cols = (
            [version_col]
            + forecast_level
            + [
                outlier_correction_col,
                outlier_lower_threshold_limit_col,
                outlier_upper_threshold_limit_col,
                outlier_method_col,
            ]
        )
        OutlierParameters = OutlierParameters[req_cols]

        req_cols = [
            version_col,
            best_fit_method_col,
            disco_period_col,
            error_metric_col,
            forecast_period_col,
            history_measure_col,
            history_period_col,
            history_time_bucket_col,
            intermittency_threshold_col,
            new_launch_period_col,
            seasonality_threshold_col,
            trend_threshold_col,
            validation_period_col,
            vol_cov_history_period_col,
            forecast_strategy_col,
        ]
        Parameters = Parameters[req_cols]

        req_cols = [
            class_col,
            version_col,
            vol_threshold_col,
            cov_threshold_col,
        ]
        SegmentationParameters = SegmentationParameters[req_cols]

        if eval(ReadFromHive):
            history_measure = "DP027" + str(Parameters[history_measure_col].iloc[0])
            # also update in Parameters dataframe
            Parameters[history_measure_col] = history_measure
        else:
            history_measure = str(Parameters[history_measure_col].iloc[0])

        logger.info("history_measure : {}".format(history_measure))

        RUN_SEGMENTATION_EVERY_CYCLE = eval(RUN_SEGMENTATION_EVERY_CYCLE)
        RUN_BEST_FIT_EVERY_CYCLE = eval(RUN_BEST_FIT_EVERY_CYCLE)
        RUN_VALIDATION_EVERY_FOLD = eval(RUN_VALIDATION_EVERY_FOLD)

        logger.info("RUN_SEGMENTATION_EVERY_CYCLE : {}".format(RUN_SEGMENTATION_EVERY_CYCLE))
        logger.info("RUN_BEST_FIT_EVERY_CYCLE : {}".format(RUN_BEST_FIT_EVERY_CYCLE))
        logger.info("RUN_VALIDATION_EVERY_FOLD : {}".format(RUN_VALIDATION_EVERY_FOLD))

        # Fill nulls in planner algo list with system algo list
        Rules[planner_algo_list_col].fillna(Rules[system_algo_list_col], inplace=True)
        # get planner rule df
        planner_rule_df = Rules[
            [
                stat_rule_col,
                planner_algo_list_col,
                planner_intermittency_col,
                planner_plc_col,
                planner_los_col,
                planner_cov_segment_col,
                planner_vol_segment_col,
                planner_trend_col,
                planner_seasonality_col,
            ]
        ]

        # fill nas in planner rule df with N/A
        planner_rule_df.fillna("N/A", inplace=True)

        if LagsToStore.upper().strip() == "ALL":
            # generate forecasts for one complete cycle
            forecast_horizon = 52 if fcst_gen_time_bucket == "Week" else 12
            LagsToStore = list(range(0, forecast_horizon))
            all_lags_flag = True
        else:
            # convert lags to store to a list
            LagsToStore = LagsToStore.split(",")
            # trim leading/trailing spaces
            LagsToStore = [int(x.strip()) for x in LagsToStore]

            # say we want to store lags 2, 4, 6 - we should generate forecast for max(2, 4, 6) + 1 - 7 cycles
            forecast_horizon = max(LagsToStore) + 1
            logger.debug(f"forecast_horizon : {forecast_horizon}")
            all_lags_flag = False

        logger.info(f"LagsToStore : {LagsToStore}")

        # retain time mapping with partial week
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()
        time_attribute_dict = {relevant_time_name: relevant_time_key}

        input_version = ForecastGenTimeBucket[version_col].unique()[0]

        current_time_period_in_relevant_bucket = CurrentTimePeriod[relevant_time_name].iloc[0]

        logger.info(
            f"current_time_period_in_relevant_bucket : {current_time_period_in_relevant_bucket}"
        )

        latest_time_name = current_time_period_in_relevant_bucket

        # adjust the latest time according to the forecast iteration's offset before getting n periods for considering history
        offset_periods = int(SellOutOffset[sell_out_offset_col].values[0])
        if offset_periods > 0:
            offset_time_periods = get_n_time_periods(
                latest_time_name,
                -offset_periods,
                relevant_time_mapping,
                time_attribute_dict,
                include_latest_value=False,
            )
            latest_time_name = offset_time_periods[0]

        logger.info(f"latest_time_name after offset {offset_periods} : {latest_time_name} ...")

        # get max value from backtestcycleperiod and supply to get n time periods
        last_n_cycles = get_n_time_periods(
            latest_value=latest_time_name,
            periods=-(max(BackTestCyclePeriods) + 1),
            time_mapping=relevant_time_mapping,
            time_attribute=time_attribute_dict,
            include_latest_value=True,
        )

        backtest_cycles = [last_n_cycles[-(i + 1)] for i in BackTestCyclePeriods]
        logger.info(f"backtest_cycles : {backtest_cycles}")

        # evaluate cycles in oldest first order
        backtest_cycles.reverse()

        backtest_cycle_df = pd.DataFrame({relevant_time_name: backtest_cycles})
        backtest_cycle_df.insert(0, version_col, input_version)

        # initialize pw col
        backtest_cycle_df[partial_week_col] = None
        backtest_cycle_df[partial_week_key_col] = None

        # Making a copy from StatLevelActual

        StatLevelActualOutputchecks = StatLevelActual.copy()

        # join actuals with pw
        StatLevelActual = StatLevelActual.merge(base_time_mapping, on=partial_week_col, how="inner")

        history_measure = "Stat Actual"

        custom_algo_list = []
        CustomL1LagBacktest = pd.DataFrame()
        rename_mapping = {}
        if BacktestOnlyCustomAlgos or (
            (not BacktestOnlyStatAlgos) and (not BacktestOnlyCustomAlgos)
        ):
            # Custom Algos Backtest Inputs
            if (
                relevant_time_name
                == o9Constants.WEEK
                # and len(CustomL1WLagBacktest) > 0
            ):
                CustomL1LagBacktest = CustomL1WLagBacktest
                backtest_algo_list = [
                    x for x in CustomL1WLagBacktest.columns if " L1 W Lag Backtest" in x
                ]
                custom_algo_list = [x.split(" L1 W Lag Backtest")[0] for x in backtest_algo_list]
            elif (
                relevant_time_name
                == o9Constants.MONTH
                # and len(CustomL1MLagBacktest) > 0
            ):
                CustomL1LagBacktest = CustomL1MLagBacktest
                backtest_algo_list = [
                    x for x in CustomL1MLagBacktest.columns if " L1 M Lag Backtest" in x
                ]
                custom_algo_list = [x.split(" L1 M Lag Backtest")[0] for x in backtest_algo_list]
            elif (
                relevant_time_name
                == o9Constants.PLANNING_MONTH
                # and len(CustomL1PMLagBacktest) > 0
            ):
                CustomL1LagBacktest = CustomL1PMLagBacktest
                backtest_algo_list = [
                    x for x in CustomL1PMLagBacktest.columns if " L1 PM Lag Backtest" in x
                ]
                custom_algo_list = [x.split(" L1 PM Lag Backtest")[0] for x in backtest_algo_list]
            elif (
                relevant_time_name
                == quarter_col
                # and len(CustomL1QLagBacktest) > 0
            ):
                CustomL1LagBacktest = CustomL1QLagBacktest
                backtest_algo_list = [
                    x for x in CustomL1QLagBacktest.columns if " L1 Q Lag Backtest" in x
                ]
                custom_algo_list = [x.split(" L1 Q Lag Backtest")[0] for x in backtest_algo_list]

            elif (
                relevant_time_name
                == planning_quarter_col
                # and len(CustomL1PQLagBacktest) > 0
            ):
                CustomL1LagBacktest = CustomL1PQLagBacktest
                backtest_algo_list = [
                    x for x in CustomL1PQLagBacktest.columns if " L1 PQ Lag Backtest" in x
                ]
                custom_algo_list = [x.split(" L1 PQ Lag Backtest")[0] for x in backtest_algo_list]
            else:
                logger.warning(f"{relevant_time_name} is not a relevant time level...")
                return (
                    Lag1FcstOutput,
                    LagNFcstOutput,
                    AllForecastWithLagDim,
                    SystemBestfitAlgorithmPlanningCycle,
                    PlanningCycleAlgoStats,
                    StabilityOutput,
                    ReasonabilityOutput,
                )

            CustomL1LagBacktest[o9Constants.PLANNING_CYCLE_DATE] = pd.to_datetime(
                CustomL1LagBacktest[o9Constants.PLANNING_CYCLE_DATE],
                errors="coerce",
            )
            # getting rename mapping for custom algos
            rename_mapping = {
                bt: stat_fcst_prefix + cu for bt, cu in zip(backtest_algo_list, custom_algo_list)
            }
            relevant_time_dim = (
                TimeDimension[[relevant_time_name, relevant_time_key]]
                .drop_duplicates()
                .rename(
                    columns={
                        relevant_time_name: "Planning Cycle Date Name",
                        relevant_time_key: o9Constants.PLANNING_CYCLE_DATE,
                    }
                )
            )
            CustomL1LagBacktest.rename(columns=rename_mapping, inplace=True)
            # if the_iteration is not None:
            #     CustomL1LagBacktest = CustomL1LagBacktest[
            #         CustomL1LagBacktest[o9Constants.FORECAST_ITERATION]
            #         == the_iteration
            #     ]
            # CustomL1LagBacktest = CustomL1LagBacktest.drop(
            #     columns=[o9Constants.FORECAST_ITERATION]
            # )
            relevant_time_dim[o9Constants.PLANNING_CYCLE_DATE] = relevant_time_dim[
                o9Constants.PLANNING_CYCLE_DATE
            ].dt.tz_localize(None)
            CustomL1LagBacktest = CustomL1LagBacktest.merge(
                relevant_time_dim,
                on=o9Constants.PLANNING_CYCLE_DATE,
                how="inner",
            )

        # adding custom algo columns
        AllForecastWithLagDim_cols = AllForecastWithLagDim_cols

        # stat item to segmentation_lob mapping
        stat_item_to_seg_lob_mapping = StatLevelActual[
            [stat_item_col, segmentation_lob_col]
        ].drop_duplicates()

        # set forecast periods here
        Parameters[forecast_period_col] = forecast_horizon

        if BestFitSelectionCriteria.empty:
            logger.warning("BestFitSelectionCriteria not populated, returning empty dataframes ...")
            return (
                Lag1FcstOutput,
                LagNFcstOutput,
                AllForecastWithLagDim,
                SystemBestfitAlgorithmPlanningCycle,
                PlanningCycleAlgoStats,
                StabilityOutput,
                ReasonabilityOutput,
            )
        bestfit_selection_criteria = BestFitSelectionCriteria[
            bestfit_selection_criteria_col
        ].unique()[0]

        # cap negatives
        StatLevelActual[history_measure] = np.where(
            StatLevelActual[history_measure] < 0,
            0,
            StatLevelActual[history_measure],
        )

        # the_iteration = StatLevelActual[
        #     o9Constants.FORECAST_ITERATION
        # ].unique()[0]

        if eval(RolloverAlgorithmflag) and CurrentCycleBestfitAlgorithm.empty:
            logger.warning(
                f"When RolloverAlgorithmflag : {RolloverAlgorithmflag}, CurrentCycleBestfitAlgorithm is a mandatory input and it is empty for {df_keys}..."
            )
            return (
                Lag1FcstOutput,
                LagNFcstOutput,
                AllForecastWithLagDim,
                SystemBestfitAlgorithmPlanningCycle,
                PlanningCycleAlgoStats,
                StabilityOutput,
                ReasonabilityOutput,
            )

        # Initialize variables to avoid referenced before assigned warning in else clause
        # StatSegmentation = pd.DataFrame()
        new_intersections = pd.DataFrame()
        CustomL1LagBacktest_current_cycle = pd.DataFrame()
        AllForecast = pd.DataFrame()
        StatSegmentation_output_checks = pd.DataFrame()
        allAlgoOutput = pd.DataFrame()
        ForecastModel = pd.DataFrame()
        ForecastData = pd.DataFrame()
        ForecastBounds = pd.DataFrame()
        AlgoListInputForForecasting = pd.DataFrame(
            columns=[x for x in CurrentCycleBestfitAlgorithm.columns]
            + [
                o9Constants.ASSIGNED_ALGORITHM_LIST,
                o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST,
                o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST,
                o9Constants.PLANNER_BESTFIT_ALGORITHM,
                o9Constants.ASSIGNED_RULE,
            ]
        )
        if len(ValidationParameters) > 0:
            ValidationPeriod = (
                ValidationParameters["Validation Period"].values[0]
                if "Validation Period" in ValidationParameters.columns
                else None
            )
            ValidationFold = (
                ValidationParameters["Validation Fold"].values[0]
                if "Validation Fold" in ValidationParameters.columns
                else None
            )
            ValidationStep = (
                ValidationParameters["Validation Step Size"].values[0]
                if "Validation Step Size" in ValidationParameters.columns
                else None
            )
        else:
            logger.warning(
                "Validation Parameters empty, please populate Validation parameters and rerun the plugin ..."
            )
            return (
                Lag1FcstOutput,
                LagNFcstOutput,
                LagFcstOutput,
                AllForecastWithLagDim,
                SystemBestfitAlgorithmPlanningCycle,
                PlanningCycleAlgoStats,
                StabilityOutput,
                ReasonabilityOutput,
            )

        if ValidationPeriod is None:
            logger.warning(
                "Validation Period input is empty, please populate Validation parameters and rerun the plugin ..."
            )
            return (
                Lag1FcstOutput,
                LagNFcstOutput,
                LagFcstOutput,
                AllForecastWithLagDim,
                SystemBestfitAlgorithmPlanningCycle,
                PlanningCycleAlgoStats,
                StabilityOutput,
                ReasonabilityOutput,
            )

        ValidationFold = ValidationFold if ValidationFold is not None else 1
        ValidationStep = ValidationStep if ValidationStep is not None else 1

        all_lag_forecast_output = []
        lag_n_forecast_output = []
        all_forecast_with_lag_dim = []
        all_systembestfitalgo_planning_cycle = []
        all_planning_cycle_algo_stats = []
        all_actual_last_n_buckets = []
        all_fcst_next_n_buckets = []

        for the_idx, the_cycle in enumerate(backtest_cycles):
            ActualLastNBuckets = pd.DataFrame()
            FcstNextNBuckets = pd.DataFrame()
            logger.info(f"Running backtest for {the_cycle}")
            the_cycle_time_period = TimeDimension[
                TimeDimension[relevant_time_name] == the_cycle
            ].sort_values(
                o9Constants.DAY_KEY
                if o9Constants.DAY_KEY in TimeDimension.columns
                else o9Constants.PARTIAL_WEEK_KEY
            )
            the_cycle_current_time_period = the_cycle_time_period.head(1).reset_index()
            logger.debug(f"Current Time period for the cycle : \n{the_cycle_current_time_period}")
            current_partial_week_key = the_cycle_current_time_period[
                o9Constants.PARTIAL_WEEK_KEY
            ].values[0]
            current_partial_week = the_cycle_current_time_period[o9Constants.PARTIAL_WEEK].values[0]

            # create the last time period dataframe - to check for new intersections
            the_last_time_period = pd.DataFrame(
                {
                    version_col: input_version,
                    relevant_time_name: the_cycle,
                },
                index=[0],
            )

            logger.debug(f"the_last_time_period\n{the_last_time_period}")

            the_planning_cycle_date = get_first_day_in_time_bucket(
                time_bucket_value=the_cycle,
                relevant_time_name=relevant_time_name,
                time_dimension=TimeDimension,
            )
            logger.debug(f"the_planning_cycle_date : {the_planning_cycle_date}")

            future_lag_periods = get_n_time_periods(
                latest_value=the_cycle,
                periods=forecast_horizon,
                time_mapping=relevant_time_mapping,
                time_attribute=time_attribute_dict,
                include_latest_value=True,
            )
            the_lag_1_time_period = future_lag_periods[1]

            filter_clause = base_time_mapping[relevant_time_name] == the_lag_1_time_period
            the_lag_1_base_time_mapping = base_time_mapping[filter_clause]
            the_lag_1_partial_weeks = list(the_lag_1_base_time_mapping[partial_week_col].unique())

            filter_clause = backtest_cycle_df[relevant_time_name] == the_cycle
            backtest_cycle_df.loc[filter_clause, partial_week_key_col] = current_partial_week_key

            the_last_time_period_with_pw_key = the_last_time_period.copy()
            the_last_time_period_with_pw_key[partial_week_key_col] = current_partial_week_key

            backtest_cycle_df.loc[filter_clause, partial_week_col] = current_partial_week

            the_history = StatLevelActual[
                StatLevelActual[partial_week_key_col].dt.tz_localize(None)
                < current_partial_week_key
            ]
            logger.debug(f"the_history, shape : {the_history.shape}")

            all_lb_cols = [stat_fcst_prefix + x + lower_bound_suffix for x in custom_algo_list]
            all_ub_cols = [stat_fcst_prefix + x + upper_bound_suffix for x in custom_algo_list]
            ActualsAndForecastData = pd.DataFrame(
                columns=[
                    o9Constants.VERSION_NAME,
                    o9Constants.SEGMENTATION_LOB,
                ]
                + forecast_level
                + [
                    o9Constants.PARTIAL_WEEK,
                    actual_cleansed_col,
                ]
                + [stat_fcst_prefix + x for x in custom_algo_list]
            )
            validation_cycles_for_this_backtest_cycle = [
                int(x)
                for x in create_planning_cycles_from_vp_vf_vs(
                    ValidationPeriod, ValidationFold, ValidationStep
                )
            ]
            planning_cycle_dates = []
            for cycle in validation_cycles_for_this_backtest_cycle:
                the_cycle_date = get_n_time_periods(
                    the_cycle_current_time_period[relevant_time_name].values[0],
                    -int(cycle),
                    TimeDimension[[relevant_time_name, relevant_time_key]].drop_duplicates(),
                    time_attribute_dict,
                    include_latest_value=False,
                )[0]
                planning_cycle_dates.append(the_cycle_date)

            logger.info(f"Planning Cycles : {planning_cycle_dates}")

            PlanningCycleDates_cycle = (
                PlanningCycleDates.merge(
                    TimeDimension,
                    left_on="Planning Cycle.[PlanningCycleDateKey]",
                    right_on=o9Constants.PARTIAL_WEEK_KEY,
                    how="inner",
                )
                .drop_duplicates(subset="Planning Cycle.[PlanningCycleDateKey]")
                .sort_values(o9Constants.PARTIAL_WEEK_KEY)
                .reset_index(drop=True)
            )
            PlanningCycleDates_cycle = PlanningCycleDates_cycle[
                PlanningCycleDates_cycle[relevant_time_name].isin(planning_cycle_dates)
            ]
            PlanningCycleDates_cycle = PlanningCycleDates_cycle[
                [o9Constants.PLANNING_CYCLE_DATE, "Planning Cycle.[PlanningCycleDateKey]"]
            ]
            AllForecastWithPC = pd.DataFrame()
            cleansed_actuals = pd.DataFrame()
            if not run_for_cycle:
                if BacktestOnlyCustomAlgos or (
                    (not BacktestOnlyStatAlgos) and (not BacktestOnlyCustomAlgos)
                ):
                    # filtering for the current cycle and current iteration
                    CustomL1LagBacktest_current_cycle = CustomL1LagBacktest[
                        CustomL1LagBacktest["Planning Cycle Date Name"] == the_cycle
                    ]
                    CustomL1LagBacktest_current_cycle = CustomL1LagBacktest_current_cycle.drop(
                        columns=[
                            o9Constants.PLANNING_CYCLE_DATE,
                            o9Constants.LAG,
                            "Planning Cycle Date Name",
                            o9Constants.VERSION_NAME,
                        ]
                    )

                if len(CustomL1LagBacktest_current_cycle) > 0:
                    # stat_bucket_weight_cml = StatBucketWeight[
                    #     StatBucketWeight[o9Constants.FORECAST_ITERATION]
                    #     == the_iteration
                    # ]
                    CustomL1LagBacktest_current_cycle = disaggregate_data(
                        source_df=CustomL1LagBacktest_current_cycle,
                        source_grain=relevant_time_name,
                        target_grain=o9Constants.PARTIAL_WEEK,
                        profile_df=StatBucketWeight.merge(
                            base_time_mapping,
                            on=o9Constants.PARTIAL_WEEK,
                            how="inner",
                        ),
                        profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                        cols_to_disaggregate=list(rename_mapping.values()),
                    )
                    CustomL1LagBacktest_current_cycle.insert(
                        loc=0,
                        column=o9Constants.VERSION_NAME,
                        value=input_version,
                    )
                    ActualsAndForecastData = the_history.merge(
                        CustomL1LagBacktest_current_cycle,
                        on=[o9Constants.VERSION_NAME] + forecast_level + [o9Constants.PARTIAL_WEEK],
                        how="outer",
                    )

                    ActualsAndForecastData.rename(
                        columns={history_measure: actual_cleansed_col},
                        inplace=True,
                    )
                    AlgoListInputForForecasting = CurrentCycleBestfitAlgorithm.merge(
                        CustomL1LagBacktest_current_cycle[
                            forecast_level + [version_col]
                        ].drop_duplicates(),
                        how="outer",
                    )
                    AlgoListInputForForecasting[o9Constants.ASSIGNED_ALGORITHM_LIST] = (
                        AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM]
                    )
                    AlgoListInputForForecasting[o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST] = (
                        AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM]
                    )
                    AlgoListInputForForecasting[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST] = ""
                    AlgoListInputForForecasting[o9Constants.PLANNER_BESTFIT_ALGORITHM] = ""
                    AlgoListInputForForecasting[o9Constants.ASSIGNED_RULE] = "Custom"
                    AlgoListInputForForecasting_cycle = AlgoListInputForForecasting.copy()

            else:
                # check if we are running the first iteration
                if the_idx == 0 or RUN_SEGMENTATION_EVERY_CYCLE:
                    StatSegmentation_copy = StatSegmentation.merge(
                        PlanningCycleDates_cycle[
                            [o9Constants.PLANNING_CYCLE_DATE]
                        ].drop_duplicates(),
                        on=o9Constants.PLANNING_CYCLE_DATE,
                        how="inner",
                    )
                    StatSegmentation_copy.columns = StatSegmentation_copy.columns.str.replace(
                        " Planning Cycle", ""
                    )
                    if len(StatSegmentation_copy) == 0:
                        logger.warning(
                            f"Segmentation results not available for the planning cycles {planning_cycle_dates}, backtest cycle - {the_cycle}, skipping this cycle..."
                        )
                        continue
                    if seasonality_l1_col in StatSegmentation_copy.columns:
                        # drop seasonality column generated from DP006 to avoid conflict with DP037
                        StatSegmentation_copy.drop(seasonality_l1_col, axis=1, inplace=True)

                    # get seasonality detection output
                    seasonality_detection_output = seasonality_detection_main(
                        Grains=Grains,
                        Actual=the_history,
                        TimeDimension=TimeDimension,
                        CurrentTimePeriod=the_cycle_current_time_period,
                        Parameters=Parameters,
                        ForecastGenTimeBucket=ForecastGenTimeBucket,
                        SegmentationOutput=StatSegmentation_copy,
                        ReadFromHive=ReadFromHive,
                        multiprocessing_num_cores=multiprocessing_num_cores,
                        df_keys=df_keys,
                        alpha=alpha,
                        lower_ci_threshold=float(ACFLowerThreshold),
                        upper_ci_threshold=float(ACFUpperThreshold),
                        skip_lags=int(ACFSkipLags),
                        diff=int(ACFDiff),
                        RequiredACFLagsInWeeks=RequiredACFLagsInWeeks,
                        SellOutOffset=SellOutOffset,
                        the_iteration=the_iteration,
                        TrendThreshold=TrendThreshold,
                    )

                    seasonality_detection_output = seasonality_detection_output[
                        req_seasonality_cols
                    ]
                    StatSegmentation_copy = StatSegmentation_copy.merge(
                        seasonality_detection_output,
                        on=forecast_level,
                        how="left",
                    )

                    logger.info("Seasonality calculation complete ...")

                    logger.info("Assigning planner assigned rules ...")

                    # keep a copy of original column so that it can be supplied to output checks
                    StatSegmentation_copy[los_l1_integer_col] = StatSegmentation_copy[los_l1_col]

                    # check if LOS is numbers
                    if StatSegmentation_copy[los_l1_col].dtype.name == "int64":
                        condition = StatSegmentation_copy[los_l1_col] < los_for_2_cycles
                        logger.debug(f"Converting {los_l1_col} into categorical values ...")
                        StatSegmentation_copy[los_l1_col] = np.where(
                            condition, "< 2 Cycles", ">= 2 Cycles"
                        )

                    logger.info("Assigning YES/NO values to trend/seasonality ...")

                    # assign yes/no to trend/seasonality/intermittency
                    StatSegmentation_copy[trend_l1_col] = np.where(
                        StatSegmentation_copy[trend_l1_col].isin(["UPWARD", "DOWNWARD"]),
                        "YES",
                        "NO",
                    )

                    StatSegmentation_copy[seasonality_l1_col] = np.where(
                        StatSegmentation_copy[seasonality_l1_col].isin(["Exists"]),
                        "YES",
                        "NO",
                    )

                    # assign rules
                    rules_assigned_the_cycle = assign_rules(
                        segmentation_output=StatSegmentation_copy,
                        rule_df=planner_rule_df,
                        column_mapping=seg_to_planner_col_mapping,
                        rule_col=stat_rule_col,
                        algo_col=planner_algo_list_col,
                        intermittent_col=intermittent_l1_col,
                        plc_col=plc_status_l1_col,
                        los_col=los_l1_col,
                        cov_segment_col=cov_segment_l1_col,
                        vol_segment_col=vol_segment_l1_col,
                        trend_col=trend_l1_col,
                        seasonality_col=seasonality_l1_col,
                    )

                    # Filter out rows where rules are not assigned
                    rules_assigned_the_cycle = rules_assigned_the_cycle[
                        rules_assigned_the_cycle[planner_algo_list_col].notna()
                    ]

                    if len(rules_assigned_the_cycle) == 0:
                        logger.warning("No rows left after filtering out NAs from rules_assigned")
                        continue
                else:
                    logger.info("Checking for new intersections added ...")

                    # check if any intersections got introduced this cycle
                    StatLevelActual[partial_week_key_col] = StatLevelActual[
                        partial_week_key_col
                    ].dt.tz_localize(None)
                    new_intersections = check_for_new_intersections(
                        the_last_time_period_with_pw_key,
                        the_idx,
                        StatLevelActual,
                        backtest_cycle_df,
                        partial_week_key_col,
                        forecast_level,
                    )

                    if len(new_intersections) > 0:
                        # join with ProductSegmentation
                        StatSegmentation_copy = new_intersections.merge(
                            StatSegmentation_copy, how="outer", on=forecast_level
                        )

                        # Assume other attributes cannot be generated and won't be used in rule assignment
                        # assign category "NEW LAUNCH"
                        StatSegmentation_copy[class_col].fillna("NEW LAUNCH", inplace=True)
                        StatSegmentation_copy[prod_cust_segment_l1_col].fillna(1.0, inplace=True)

                        # TODO : Need to calculate the other attributes for new intersections
                if len(OutlierThresholds) > 0:
                    OutlierAbsoluteThreshold = OutlierThresholds[
                        [o9Constants.VERSION_NAME, "Outlier Absolute Threshold"]
                    ]
                    OutlierPercentageThreshold = OutlierThresholds[
                        [o9Constants.VERSION_NAME, "Outlier Percentage Threshold"]
                    ]
                else:
                    OutlierAbsoluteThreshold = pd.DataFrame()
                    OutlierPercentageThreshold = pd.DataFrame()

                Actual_outlier_input = the_history[
                    forecast_level
                    + [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK, history_measure]
                ]
                Actual_outlier_input[o9Constants.SLICE_ASSOCIATION_STAT] = 1
                # perform outlier correction
                cleansed_actuals, actual_l1_output = outlier_correction_main(
                    Grains=Grains,
                    ReadFromHive=ReadFromHive,
                    Actual=Actual_outlier_input,
                    CurrentTimePeriod=the_cycle_current_time_period,
                    HistoryPeriod=Parameters,
                    OutlierParameters=OutlierParameters,
                    TimeDimension=TimeDimension,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    StatBucketWeight=StatBucketWeight,
                    SellOutOffset=SellOutOffset,
                    the_iteration=the_iteration,
                    multiprocessing_num_cores=multiprocessing_num_cores,
                    df_keys=df_keys,
                    smooth_fraction=float(smooth_fraction),
                    OutlierAbsoluteThreshold=OutlierAbsoluteThreshold,
                    OutlierPercentageThreshold=OutlierPercentageThreshold,
                )

                if len(cleansed_actuals) == 0:
                    logger.warning(
                        "cleansed_actuals is empty for backtest period : {}, slice : {}".format(
                            the_cycle, df_keys
                        )
                    )
                    continue

                cleansed_actuals = cleansed_actuals[req_cols_from_outlier_output]

                logger.info("Generating forecasts ...")

                non_disc_intersections = StatSegmentation_copy[non_disc_intersections_req_cols]

                filter_clause = non_disc_intersections[class_col] != "DISC"
                the_non_disc_intersections = StatSegmentation_copy[filter_clause]

                if eval(RolloverAlgorithmflag):
                    # check for first cycle
                    if the_idx == 0:
                        # generate forecasts for only the best fit algo from current cycle
                        AlgoListInputForForecasting = rules_assigned_the_cycle.merge(
                            CurrentCycleBestfitAlgorithm,
                            on=[version_col] + forecast_level,
                            how="inner",
                        )
                        # getting ensemble algorithms to use if bestfit is Ensemble
                        AlgoListInputForForecasting = AlgoListInputForForecasting.merge(
                            SystemEnsembleAlgorithmList,
                            on=[version_col] + forecast_level,
                            how="left",
                        )

                        AlgoListInputForForecasting[o9Constants.ASSIGNED_ALGORITHM_LIST] = np.where(
                            AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM]
                            == "Ensemble",
                            AlgoListInputForForecasting[system_ensemble_algo_list_col],
                            AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM],
                        )

                        AlgoListInputForForecasting[o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST] = (
                            np.where(
                                AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM]
                                == "Ensemble",
                                AlgoListInputForForecasting[system_ensemble_algo_list_col],
                                AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM],
                            )
                        )

                        AlgoListInputForForecasting[o9Constants.PLANNER_BESTFIT_ALGORITHM] = (
                            np.where(
                                AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM]
                                == "Ensemble",
                                AlgoListInputForForecasting[system_ensemble_algo_list_col],
                                AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM],
                            )
                        )

                        AlgoListInputForForecasting[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST] = (
                            np.where(
                                AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM]
                                == "Ensemble",
                                AlgoListInputForForecasting[system_ensemble_algo_list_col],
                                AlgoListInputForForecasting[o9Constants.BESTFIT_ALGORITHM],
                            )
                        )

                        AlgoListInputForForecasting[o9Constants.ASSIGNED_RULE] = (
                            AlgoListInputForForecasting[o9Constants.STAT_RULE]
                        )

                elif the_idx == 0 or RUN_BEST_FIT_EVERY_CYCLE:
                    AlgoListInputForForecasting = (
                        rules_assigned_the_cycle[
                            [version_col]
                            + forecast_level
                            + [
                                o9Constants.STAT_RULE,
                                planner_algo_list_col,
                                trend_l1_col,
                                seasonality_l1_col,
                            ]
                        ]
                        .drop_duplicates()
                        .copy()
                    )
                    AlgoListInputForForecasting.rename(
                        columns={
                            planner_algo_list_col: o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST,
                            o9Constants.STAT_RULE: o9Constants.ASSIGNED_RULE,
                        },
                        inplace=True,
                    )
                    AlgoListInputForForecasting[o9Constants.ASSIGNED_ALGORITHM_LIST] = (
                        AlgoListInputForForecasting[o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST]
                    )

                    # add dummy columns
                    AlgoListInputForForecasting[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST] = (
                        np.nan
                    )
                    AlgoListInputForForecasting[o9Constants.PLANNER_BESTFIT_ALGORITHM] = np.nan
                    AlgoListInputForForecasting[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST] = (
                        np.nan
                    )
                else:
                    # if idx > 1, there's no need to join on best fit dataframe since it was already joined in previous cycle run
                    pass

                # SCHM Logic for backtesting
                # Renaming columns as per backtesting logic
                SeasonalIndices = SeasonalIndices.rename(
                    columns={
                        validation_seasonal_index_backtest: validation_seasonal_index,
                        seasonal_index_backtest: seasonal_index,
                    }
                )

                logger.info("SeasonalIndices : {}".format(SeasonalIndices))

                cleansed_actuals = cleansed_actuals.merge(
                    StatLevelActual[
                        [
                            o9Constants.VERSION_NAME,
                            o9Constants.PARTIAL_WEEK,
                            o9Constants.STAT_ACTUAL,
                        ]
                        + forecast_level
                    ],
                    on=[o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + forecast_level,
                    how="left",
                )

                cleansed_actuals[o9Constants.STAT_ACTUAL].fillna(0, inplace=True)

                cleansed_actuals = cleansed_actuals.rename(
                    columns={actual_cleansed_system_col: actual_cleansed_col}
                )
                AllForecastWithPC = validation_fcst_main(
                    Grains=Grains,
                    IncludeDiscIntersections=IncludeDiscIntersections,
                    RUN_VALIDATION_EVERY_FOLD=RUN_VALIDATION_EVERY_FOLD,
                    Actual=cleansed_actuals,
                    StatSegment=the_non_disc_intersections,
                    Parameters=Parameters,
                    AlgoParameters=AlgoParameters,
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=the_cycle_current_time_period,
                    AlgoList=AlgoListInputForForecasting,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    StatBucketWeight=StatBucketWeight,
                    MasterAlgoList=MasterAlgoList,
                    DefaultAlgoParameters=DefaultAlgoParameters,
                    PlannerOverrideCycles="None",
                    SeasonalIndices=SeasonalIndices,
                    ForecastEngine=ForecastEngine,
                    SellOutOffset=SellOutOffset,
                    multiprocessing_num_cores=multiprocessing_num_cores,
                    PlanningCycleDates=PlanningCycleDates,
                    ValidationParameters=ValidationParameters,
                    history_measure=HistoryMeasure,
                    the_iteration=the_iteration,
                    df_keys=df_keys,
                )
                ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION] = the_iteration
                AllForecast, ForecastModel = forecasting_main(
                    Grains=Grains,
                    history_measure=HistoryMeasure,
                    AlgoList=AlgoListInputForForecasting,
                    Actual=cleansed_actuals,
                    TimeDimension=TimeDimension,
                    ForecastParameters=Parameters,
                    AlgoParameters=AlgoParameters,
                    CurrentTimePeriod=the_cycle_current_time_period,
                    non_disc_intersections=the_non_disc_intersections,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    StatBucketWeight=StatBucketWeight,
                    UseHolidays=UseHolidays,
                    SellOutOffset=SellOutOffset,
                    IncludeDiscIntersections=IncludeDiscIntersections,
                    multiprocessing_num_cores=multiprocessing_num_cores,
                    df_keys=df_keys,
                    MasterAlgoList=MasterAlgoList,
                    DefaultAlgoParameters=DefaultAlgoParameters,
                    SeasonalIndices=SeasonalIndices,
                    model_params=False,
                )

                if len(AllForecast) == 0:
                    logger.warning(
                        "AllForecast is empty for backtest period : {}, slice : {}".format(
                            the_cycle, df_keys
                        )
                    )
                    continue

                CustomL1LagBacktest_current_cycle = pd.DataFrame()
                AlgoListInputForForecasting_cycle = AlgoListInputForForecasting.copy()
                if BacktestOnlyCustomAlgos or (
                    (not BacktestOnlyStatAlgos) and (not BacktestOnlyCustomAlgos)
                ):

                    # filtering for the current cycle and current iteration
                    CustomL1LagBacktest_current_cycle = CustomL1LagBacktest[
                        CustomL1LagBacktest["Planning Cycle Date Name"] == the_cycle
                    ]
                    CustomL1LagBacktest_current_cycle = CustomL1LagBacktest_current_cycle.drop(
                        columns=[
                            o9Constants.PLANNING_CYCLE_DATE,
                            o9Constants.LAG,
                            "Planning Cycle Date Name",
                            o9Constants.VERSION_NAME,
                        ]
                    )

                if len(CustomL1LagBacktest_current_cycle) > 0:
                    # stat_bucket_weight_cml = StatBucketWeight[
                    #     StatBucketWeight[o9Constants.FORECAST_ITERATION]
                    #     == the_iteration
                    # ]
                    CustomL1LagBacktest_current_cycle = disaggregate_data(
                        source_df=CustomL1LagBacktest_current_cycle,
                        source_grain=relevant_time_name,
                        target_grain=o9Constants.PARTIAL_WEEK,
                        profile_df=StatBucketWeight.merge(
                            base_time_mapping,
                            on=o9Constants.PARTIAL_WEEK,
                            how="inner",
                        ),
                        profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                        cols_to_disaggregate=list(rename_mapping.values()),
                    )
                    CustomL1LagBacktest_current_cycle.insert(
                        loc=0,
                        column=o9Constants.VERSION_NAME,
                        value=input_version,
                    )
                    if len(AlgoListInputForForecasting) > 0:
                        AlgoListInputForForecasting_cycle[
                            o9Constants.ASSIGNED_ALGORITHM_LIST
                        ] += "," + ",".join(custom_algo_list)

                    AlgoListInputForForecasting_cycle = AlgoListInputForForecasting_cycle.merge(
                        CustomL1LagBacktest_current_cycle[
                            forecast_level + [version_col]
                        ].drop_duplicates(),
                        how="outer",
                    )
                    AlgoListInputForForecasting_cycle[o9Constants.ASSIGNED_ALGORITHM_LIST].fillna(
                        ",".join(custom_algo_list), inplace=True
                    )
                    AlgoListInputForForecasting_cycle[o9Constants.ASSIGNED_RULE].fillna(
                        "Custom", inplace=True
                    )

                    # need to add custom algos values to AllForecast for identify_best_fit
                    AllForecast = AllForecast.merge(
                        CustomL1LagBacktest_current_cycle,
                        on=forecast_level + [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK],
                        how="outer",
                    )
                    all_lb_cols = [
                        stat_fcst_prefix + x + lower_bound_suffix for x in custom_algo_list
                    ]
                    all_ub_cols = [
                        stat_fcst_prefix + x + upper_bound_suffix for x in custom_algo_list
                    ]
                    for col in all_lb_cols + all_ub_cols:
                        AllForecast[col] = np.nan
                    RolloverAlgorithmflag = "False"
                # create column and assign dummy value so that output check doesn't fail
                AllForecast_copy = AllForecast.copy()
                AllForecast[o9Constants.STAT_FCST_L1] = np.nan
                AllForecast[PLANNER_STAT_FORECAST] = np.nan
                AllForecast[STAT_FCST_L1_LC] = np.nan
                StatSegmentation_output_checks = StatSegmentation_copy.copy()
                StatSegmentation_output_checks[los_l1_col] = StatSegmentation_output_checks[
                    los_l1_integer_col
                ]
                StatSegmentation_output_checks = StatSegmentation_output_checks.merge(
                    AlgoListInputForForecasting_cycle[
                        forecast_level
                        + [
                            o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST,
                            o9Constants.ASSIGNED_RULE,
                            o9Constants.ASSIGNED_ALGORITHM_LIST,
                            o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST,
                            o9Constants.PLANNER_BESTFIT_ALGORITHM,
                        ]
                    ],
                    on=forecast_level,
                    how="left",
                )
                StatSegmentation_output_checks[o9Constants.BESTFIT_ALGORITHM] = np.nan
                logger.info("Generating best fit ...")

                cleansed_actuals.rename(
                    columns={actual_cleansed_system_col: actual_cleansed_col},
                    inplace=True,
                )
                ActualsAndForecastData = AllForecast_copy.drop(
                    ALL_STAT_FCST_BOUND_COLUMNS, axis=1, errors="ignore"
                ).merge(cleansed_actuals, on=join_cols, how="outer")

                ForecastBounds = AllForecast_copy[ForecastBounds_req_cols]

                AllForecast.drop(
                    columns=ALL_STAT_FCST_BOUND_COLUMNS, axis=1, inplace=True, errors="ignore"
                )

                # Including Stat Actual
                cleansed_actuals = cleansed_actuals.merge(
                    StatLevelActualOutputchecks.drop(columns=segmentation_lob_col), how="left"
                )

                # add dummy values to mandatory columns
                ForecastModel[o9Constants.VALIDATION_ERROR] = np.nan
                ForecastModel[o9Constants.COMPOSITE_ERROR] = np.nan

                (
                    allAlgoOutput,
                    bestFitOutput,
                    ActualLastNBuckets,
                    FcstNextNBuckets,
                    algoStatsForBestFitMember,
                ) = output_checks(
                    Grains=Grains,
                    TimeDimension=TimeDimension,
                    Actual=cleansed_actuals,
                    ForecastData=AllForecast,
                    SegmentationOutput=StatSegmentation_output_checks,
                    CurrentTimePeriod=the_cycle_current_time_period,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    AlgoStats=ForecastModel,
                    ForecastSetupConfiguration=Parameters,
                    TrendVariationThreshold=float(TrendVariationThreshold),
                    LevelVariationThreshold=float(LevelVariationThreshold),
                    RangeVariationThreshold=float(RangeVariationThreshold),
                    SeasonalVariationThreshold=float(SeasonalVariationThreshold),
                    SeasonalVariationCountThreshold=float(SeasonalVariationCountThreshold),
                    MinimumIndicePercentage=float(MinimumIndicePercentage),
                    AbsTolerance=float(AbsTolerance),
                    COCCVariationThreshold=float(COCCVariationThreshold),
                    SellOutOffset=SellOutOffset,
                    ReasonabilityCycles=float(ReasonabilityCycles),
                    df_keys=df_keys,
                )
                if bestfit_selection_criteria == "Validation Error plus Violations":
                    # add segmentation lob column
                    ActualsAndForecastData = ActualsAndForecastData.merge(
                        stat_item_to_seg_lob_mapping,
                        on=stat_item_col,
                        how="inner",
                    )

                    allAlgoOutput = allAlgoOutput.merge(
                        stat_item_to_seg_lob_mapping,
                        on=stat_item_col,
                        how="inner",
                    )
                else:
                    allAlgoOutput = pd.DataFrame()

            # Ensure AllForecastWithPC is defined and not empty before merging
            if AllForecastWithPC is None or AllForecastWithPC.empty:
                if StatFcstCMLPlanningCycle is not None and not StatFcstCMLPlanningCycle.empty:
                    AllForecastWithPC = StatFcstCMLPlanningCycle.copy()
            else:
                AllForecastWithPC = AllForecastWithPC.merge(
                    StatFcstCMLPlanningCycle,
                    on=forecast_level
                    + [
                        o9Constants.VERSION_NAME,
                        o9Constants.PLANNING_CYCLE_DATE,
                        o9Constants.PARTIAL_WEEK,
                    ],
                    how="left",
                )

            # Identify best fit model
            ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION] = the_iteration
            BestFitSelectionCriteria[o9Constants.FORECAST_ITERATION] = the_iteration
            Parameters[o9Constants.FORECAST_ITERATION] = the_iteration
            cleansed_actuals[o9Constants.FORECAST_ITERATION] = the_iteration
            AllForecastWithPC[o9Constants.FORECAST_ITERATION] = the_iteration
            AlgoListInputForForecasting_identifybestfit = AlgoListInputForForecasting_cycle.copy()
            AlgoListInputForForecasting_identifybestfit[o9Constants.FORECAST_ITERATION] = (
                the_iteration
            )
            ValidationParameters[o9Constants.FORECAST_ITERATION] = the_iteration
            Weights[o9Constants.FORECAST_ITERATION] = the_iteration
            ForecastEngine[o9Constants.FORECAST_ITERATION] = the_iteration
            allAlgoOutput[o9Constants.FORECAST_ITERATION] = the_iteration
            BestFitAlgo, ValidationError, ValidationErrorPlanningCycle, ValidationForecast = (
                identify_best_fit(
                    Grains=Grains,
                    HistoryMeasure=HistoryMeasure,
                    TimeDimension=TimeDimension,
                    ForecastParameters=Parameters,
                    PlanningCycleDates=PlanningCycleDates,
                    CurrentTimePeriod=the_cycle_current_time_period,
                    Actuals=cleansed_actuals,
                    ForecastData=AllForecastWithPC,
                    ValidationParameters=ValidationParameters,
                    OverrideFlatLineForecasts=OverrideFlatLineForecasts,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    AssignedAlgoList=AlgoListInputForForecasting_identifybestfit,
                    SelectionCriteria=BestFitSelectionCriteria,
                    MasterAlgoList=MasterAlgoList,
                    Weights=Weights,
                    Violations=allAlgoOutput,
                    SellOutOffset=SellOutOffset,
                    multiprocessing_num_cores=multiprocessing_num_cores,
                    ForecastEngine=ForecastEngine,
                    df_keys=df_keys,
                )
            )

            validation_method = Parameters["Bestfit Method"].iloc[0]
            # getting validation method

            ValidationError[o9Constants.VALIDATION_METHOD] = validation_method
            if AllForecast.empty:
                AllForecast = CustomL1LagBacktest_current_cycle.copy()
                for col in all_lb_cols + all_ub_cols:
                    AllForecast[col] = np.nan
            # create column and assign dummy value so that output check doesn't fail
            AllForecast[system_stat_fcst_l1_col] = np.nan
            AllForecast[o9Constants.STAT_FCST_L1] = np.nan
            AllForecast[PLANNER_STAT_FORECAST] = np.nan
            AllForecast[STAT_FCST_L1_LC] = np.nan
            if StatSegmentation_output_checks.empty:
                StatSegmentation_output_checks = BestFitAlgo.copy()
                StatSegmentation_output_checks[o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST] = (
                    ",".join(custom_algo_list)
                )
                StatSegmentation_output_checks[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST] = (
                    ",".join(custom_algo_list)
                )
                StatSegmentation_output_checks = StatSegmentation_output_checks.merge(
                    ValidationError[forecast_level + [o9Constants.STAT_RULE]].drop_duplicates()
                )
                StatSegmentation_output_checks.rename(
                    columns={o9Constants.STAT_RULE: o9Constants.ASSIGNED_RULE},
                    inplace=True,
                )
                StatSegmentation_output_checks[planner_best_fit_algo_col] = (
                    StatSegmentation_output_checks[system_best_fit_algo_col]
                )
                StatSegmentation_output_checks[o9Constants.BESTFIT_ALGORITHM] = (
                    StatSegmentation_output_checks[system_best_fit_algo_col]
                )
                StatSegmentation_output_checks[los_l1_col] = ""
                StatSegmentation_output_checks[seasonality_l1_col] = "Exists"
                StatSegmentation_output_checks[plc_status_l1_col] = ""
                StatSegmentation_output_checks[intermittent_l1_col] = ""

            forecast_strategy = Parameters["Forecast Strategy"].iloc[0]
            if forecast_strategy == "Bestfit":
                BestFitAlgo_Ensemble = pd.DataFrame(columns=BestFitAlgo.columns)
                ValidationError_Ensemble = pd.DataFrame(columns=ValidationError.columns)

                cols_required_in_all_ensemble_desc_df = (
                    [o9Constants.VERSION_NAME]
                    + forecast_level
                    + [
                        o9Constants.STAT_ALGORITHM,
                        o9Constants.STAT_RULE,
                        o9Constants.ALGORITHM_PARAMETERS,
                        o9Constants.RUN_TIME,
                    ]
                )
                AllEnsembleDesc_Ensemble = pd.DataFrame(
                    columns=cols_required_in_all_ensemble_desc_df
                )

                cols_required_in_all_ensemble_fcst_df = (
                    [o9Constants.VERSION_NAME]
                    + forecast_level
                    + [
                        partial_week_col,
                        "Stat Fcst Ensemble",
                    ]
                )
                AllEnsembleFcst_Ensemble = pd.DataFrame(
                    columns=cols_required_in_all_ensemble_fcst_df
                )

                cols_required_in_system_ensemble_algo_list = (
                    [o9Constants.VERSION_NAME]
                    + forecast_level
                    + [
                        system_ensemble_algo_list_col,
                    ]
                )
                SystemEnsembleAlgorithmList_Ensemble = pd.DataFrame(
                    columns=cols_required_in_system_ensemble_algo_list
                )
                FcstNextNBuckets_Ensemble = pd.DataFrame()
                OutputAllAlgo_Ensemble = pd.DataFrame()
            else:
                # Adding Stat Actual

                ActualsAndForecastDatawithStatActual = ActualsAndForecastData.merge(
                    StatLevelActualOutputchecks.drop(columns=segmentation_lob_col), how="left"
                )

                (
                    BestFitAlgo_Ensemble,
                    ValidationError_Ensemble,
                    FcstNextNBuckets_Ensemble,
                    AllEnsembleDesc_Ensemble,
                    AllEnsembleFcst_Ensemble,
                    SystemEnsembleAlgorithmList_Ensemble,
                    OutputAllAlgo_Ensemble,
                ) = calculate_ensemble_fcst(
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=the_cycle_current_time_period,
                    ForecastParameters=Parameters,
                    ActualsAndForecastData=ActualsAndForecastDatawithStatActual,
                    PlanningCycleDates=PlanningCycleDates,
                    ValidationParameters=ValidationParameters,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    AssignedAlgoList=AlgoListInputForForecasting_cycle.copy(),
                    Weights=Weights,
                    SelectionCriteria=BestFitSelectionCriteria,
                    Violations=allAlgoOutput,
                    EnsembleParameters=EnsembleParameters,
                    IncludeEnsemble=IncludeEnsemble,
                    EnsembleWeights=EnsembleWeights,
                    AlgoStats=ValidationError,
                    MasterAlgoList=MasterAlgoList,
                    ForecastData=AllForecast,
                    SegmentationOutput=StatSegmentation_output_checks,
                    StatBucketWeight=StatBucketWeight,
                    OverrideFlatLineForecasts=OverrideFlatLineForecasts,
                    EnsembleOnlyStatAlgos=BacktestOnlyStatAlgos,
                    EnsembleOnlyCustomAlgos=BacktestOnlyCustomAlgos,
                    Grains=Grains,
                    HistoryMeasure=HistoryMeasure,
                    multiprocessing_num_cores=multiprocessing_num_cores,
                    TrendVariationThreshold=float(TrendVariationThreshold),
                    LevelVariationThreshold=float(LevelVariationThreshold),
                    RangeVariationThreshold=float(RangeVariationThreshold),
                    SeasonalVariationThreshold=float(SeasonalVariationThreshold),
                    SeasonalVariationCountThreshold=float(SeasonalVariationCountThreshold),
                    ReasonabilityCycles=float(ReasonabilityCycles),
                    MinimumIndicePercentage=float(MinimumIndicePercentage),
                    AbsTolerance=float(AbsTolerance),
                    COCCVariationThreshold=float(COCCVariationThreshold),
                    the_iteration=the_iteration,
                    df_keys=df_keys,
                    ForecastEngine=ForecastEngine,
                )

            # get planning cycle algo stats
            AllEnsembleDesc_Ensemble[o9Constants.VALIDATION_METHOD] = validation_method
            intersections_present_in_forecast_model_ensemble = AllEnsembleDesc_Ensemble[
                forecast_level
            ].drop_duplicates()

            if ForecastModel.empty:
                ForecastModel = pd.DataFrame(columns=AllEnsembleDesc_Ensemble.columns)

            ForecastModel = ForecastModel.merge(
                intersections_present_in_forecast_model_ensemble,
                how="outer",
                indicator=True,
            )
            ForecastModel = ForecastModel[ForecastModel["_merge"] == "left_only"]
            ForecastModel = pd.concat([ForecastModel, AllEnsembleDesc_Ensemble])
            ForecastModel.drop(columns=["_merge"], axis=1, inplace=True)

            the_planning_cycle_algo_stats = ForecastModel.copy()
            the_planning_cycle_algo_stats.drop(
                [
                    o9Constants.VALIDATION_ERROR,
                    o9Constants.COMPOSITE_ERROR,
                ],
                axis=1,
                inplace=True,
                errors="ignore",
            )
            the_planning_cycle_algo_stats[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date
            # the_planning_cycle_algo_stats[
            #     o9Constants.FORECAST_ITERATION
            # ] = the_iteration

            the_planning_cycle_algo_stats.rename(
                columns={
                    o9Constants.VALIDATION_METHOD: o9Constants.VALIDATION_METHOD_PLANNING_CYCLE,
                    o9Constants.ALGORITHM_PARAMETERS: o9Constants.ALGORITHM_PARAMETERS_PLANNING_CYCLE,
                },
                inplace=True,
            )

            # Add validation error
            # remove common intersections and consider ensemble ones
            intersections_present_in_ensemble = ValidationError_Ensemble[
                forecast_level
            ].drop_duplicates()
            ValidationError = ValidationError.merge(
                intersections_present_in_ensemble,
                how="outer",
                indicator=True,
            )
            ValidationError = ValidationError[ValidationError["_merge"] == "left_only"]

            ValidationError = pd.concat([ValidationError, ValidationError_Ensemble])

            the_cycle_validation_error = ValidationError[
                forecast_level
                + [
                    o9Constants.STAT_ALGORITHM,
                    o9Constants.VALIDATION_ERROR,
                    o9Constants.COMPOSITE_ERROR,
                ]
            ].drop_duplicates()

            # merge with algo parameters and validation method
            the_planning_cycle_algo_stats = the_planning_cycle_algo_stats.merge(
                the_cycle_validation_error,
                on=forecast_level + [stat_algorithm_col],
                how="outer",
            )
            the_planning_cycle_custom_algo_stats = the_cycle_validation_error[
                the_cycle_validation_error[stat_algorithm_col].isin(custom_algo_list)
            ]
            the_planning_cycle_algo_stats = pd.concat(
                [
                    the_planning_cycle_algo_stats,
                    the_planning_cycle_custom_algo_stats,
                ]
            )
            # filling nulls
            the_planning_cycle_algo_stats[version_col] = the_planning_cycle_algo_stats[
                version_col
            ].fillna(input_version)
            the_planning_cycle_algo_stats[o9Constants.PLANNING_CYCLE_DATE] = (
                the_planning_cycle_algo_stats[o9Constants.PLANNING_CYCLE_DATE].fillna(
                    the_planning_cycle_date
                )
            )
            the_planning_cycle_algo_stats[o9Constants.VALIDATION_METHOD_PLANNING_CYCLE] = (
                the_planning_cycle_algo_stats[o9Constants.VALIDATION_METHOD_PLANNING_CYCLE].fillna(
                    validation_method
                )
            )
            the_planning_cycle_algo_stats[o9Constants.ALGORITHM_PARAMETERS_PLANNING_CYCLE] = (
                the_planning_cycle_algo_stats[
                    o9Constants.ALGORITHM_PARAMETERS_PLANNING_CYCLE
                ].fillna("")
            )

            # rename/add column
            the_planning_cycle_algo_stats[o9Constants.VALIDATION_ERROR_PLANNING_CYCLE] = (
                the_planning_cycle_algo_stats[o9Constants.VALIDATION_ERROR]
            )

            the_planning_cycle_algo_stats[o9Constants.COMPOSITE_ERROR_PLANNING_CYCLE] = (
                the_planning_cycle_algo_stats[o9Constants.COMPOSITE_ERROR]
            )

            the_planning_cycle_algo_stats = the_planning_cycle_algo_stats[
                PlanningCycleAlgoStats_cols
            ]
            the_planning_cycle_algo_stats[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date
            all_planning_cycle_algo_stats.append(the_planning_cycle_algo_stats)

            # remove common intersections and consider ensemble ones
            intersections_present_in_bestfit_ensemble = BestFitAlgo_Ensemble[
                forecast_level
            ].drop_duplicates()
            BestFitAlgo = BestFitAlgo.merge(
                intersections_present_in_bestfit_ensemble,
                how="outer",
                indicator=True,
            )
            BestFitAlgo = BestFitAlgo[BestFitAlgo["_merge"] == "left_only"]
            BestFitAlgo = pd.concat([BestFitAlgo, BestFitAlgo_Ensemble])
            BestFitAlgo.drop(columns=["_merge"], axis=1, inplace=True)

            if eval(RolloverAlgorithmflag):
                the_cycle_system_bestfit_algo = CurrentCycleBestfitAlgorithm.copy()
                the_cycle_system_bestfit_algo[
                    o9Constants.SYSTEM_BESTFIT_ALGORITHM_PLANNING_CYCLE
                ] = the_cycle_system_bestfit_algo[o9Constants.BESTFIT_ALGORITHM]
            else:
                the_cycle_system_bestfit_algo = BestFitAlgo[
                    [version_col] + forecast_level + [system_best_fit_algo_col]
                ]
                the_cycle_system_bestfit_algo[
                    o9Constants.SYSTEM_BESTFIT_ALGORITHM_PLANNING_CYCLE
                ] = the_cycle_system_bestfit_algo[system_best_fit_algo_col]
            the_cycle_system_bestfit_algo[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date
            # the_cycle_system_bestfit_algo[
            #     o9Constants.FORECAST_ITERATION
            # ] = the_iteration

            the_cycle_system_bestfit_algo = the_cycle_system_bestfit_algo[
                SystemBestfitAlgorithmPlanningCycle_cols
            ]

            all_systembestfitalgo_planning_cycle.append(the_cycle_system_bestfit_algo)

            if actual_cleansed_system_col in ActualsAndForecastData.columns:
                ForecastData = ActualsAndForecastData.drop(
                    actual_cleansed_system_col,
                    axis=1,
                )

            if actual_cleansed_col in ActualsAndForecastData.columns:
                ForecastData = ActualsAndForecastData.drop(
                    actual_cleansed_col,
                    axis=1,
                )

            if eval(RolloverAlgorithmflag):
                BestFitAlgoInputForPopulateBestFit = CurrentCycleBestfitAlgorithm.copy()
                BestFitAlgoInputForPopulateBestFit[system_best_fit_algo_col] = (
                    BestFitAlgoInputForPopulateBestFit[o9Constants.BESTFIT_ALGORITHM]
                )
                BestFitAlgoInputForPopulateBestFit[o9Constants.PLANNER_BESTFIT_ALGORITHM] = (
                    BestFitAlgoInputForPopulateBestFit[o9Constants.BESTFIT_ALGORITHM]
                )
            else:
                BestFitAlgoInputForPopulateBestFit = BestFitAlgo[
                    [version_col] + forecast_level + [system_best_fit_algo_col]
                ]
                # populate planner best fit algo with same data
                BestFitAlgoInputForPopulateBestFit[planner_best_fit_algo_col] = (
                    BestFitAlgoInputForPopulateBestFit[system_best_fit_algo_col]
                )

            if allAlgoOutput.empty:
                ViolationsInput = pd.DataFrame()
            else:
                ViolationsInput = allAlgoOutput.merge(
                    ForecastModel,
                    on=[version_col] + forecast_level + [stat_algorithm_col, stat_rule_col],
                    how="inner",
                )

                # add dummy values to mandatory columns
                ViolationsInput[validation_actual_col] = np.nan
                ViolationsInput[validation_fcst_col] = np.nan
                ViolationsInput[composite_error_col] = np.nan
                ViolationsInput[validation_fcst_abs_error_col] = np.nan
                ViolationsInput[validation_error_col] = np.nan
                ViolationsInput[fcst_next_n_buckets] = np.nan

            if ForecastBounds.empty:
                if AllForecast.empty:
                    ForecastBounds = pd.DataFrame(
                        columns=(
                            [version_col]
                            + forecast_level
                            + [partial_week_col]
                            + all_ub_cols
                            + all_lb_cols
                        )
                    )
                else:
                    ForecastBounds = AllForecast[
                        [version_col]
                        + forecast_level
                        + [partial_week_col]
                        + all_ub_cols
                        + all_lb_cols
                    ].drop_duplicates()

            if not run_for_cycle:
                if AllForecast.empty:
                    ForecastData = pd.DataFrame(
                        columns=(
                            [version_col]
                            + forecast_level
                            + [partial_week_col]
                            + [stat_fcst_prefix + x for x in custom_algo_list]
                            + all_lb_cols
                            + all_ub_cols
                        )
                    )
                else:
                    ForecastData = AllForecast[
                        [version_col]
                        + forecast_level
                        + [partial_week_col]
                        + [stat_fcst_prefix + x for x in custom_algo_list]
                        + all_lb_cols
                        + all_ub_cols
                    ]

            if not AllEnsembleFcst_Ensemble.empty:
                # in case ensemble output is empty
                AllEnsembleFcst_Ensemble["Stat Fcst Ensemble"] = AllEnsembleFcst_Ensemble[
                    "Stat Fcst Ensemble"
                ].astype("float")
                # getting forecast results for ensemble
                ForecastData = ForecastData.merge(AllEnsembleFcst_Ensemble, how="outer")

            # populate best fit predictions
            (
                BestFitForecast,
                BestFitAlgorithmCandidateOutput,
                BestFitViolationOutput,
            ) = populate_best_fit_forecast(
                Grains=Grains,
                TimeDimension=TimeDimension,
                ForecastParameters=Parameters,
                CurrentTimePeriod=the_cycle_current_time_period,
                ForecastData=ForecastData,
                ForecastBounds=ForecastBounds,
                BestFitAlgo=BestFitAlgoInputForPopulateBestFit,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                StatBucketWeight=StatBucketWeight,
                Violations=ViolationsInput,
                df_keys=df_keys,
            )

            if len(BestFitForecast) == 0:
                logger.warning(
                    "AllForecast is empty for backtest period : {}, slice : {}".format(
                        the_cycle, df_keys
                    )
                )
                continue

            BestFitForecast = BestFitForecast.drop_duplicates(
                subset=forecast_level + [partial_week_col]
            )

            # Filter forecasts for only the specified lag
            the_lag_1_output = BestFitForecast[
                BestFitForecast[partial_week_col].isin(the_lag_1_partial_weeks)
            ]

            # rename column
            the_lag_1_output.rename(
                columns={system_stat_fcst_l1_col: stat_fcst_l1_lag1_col},
                inplace=True,
            )

            # # add iteration
            the_lag_1_output[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date

            # select required columns from output, rename
            the_lag_1_output = the_lag_1_output[Lag1FcstOutput_cols]

            logger.info("the_lag_1_output, shape : {the_lag_1_output.shape}")

            logger.info("Appending results to master list ...")
            all_lag_forecast_output.append(the_lag_1_output)

            logger.debug("Creating lag mapping ...")

            # adding lag dimension to AllForecast and BestFitForecast
            # generate relevant month to lag mapping
            the_lag_mapping = pd.DataFrame(future_lag_periods, columns=[relevant_time_name])
            the_lag_mapping[lag_col] = the_lag_mapping.index

            # typecasting the column to make sure there are no decimals in the lag col
            the_lag_mapping[lag_col] = the_lag_mapping[lag_col].astype(int)

            filter_clause = the_lag_mapping[lag_col].isin(LagsToStore)
            the_lag_mapping = the_lag_mapping[filter_clause]

            lag_n_mapping = the_lag_mapping.merge(
                TimeDimension[[o9Constants.PARTIAL_WEEK, relevant_time_name]].drop_duplicates(),
                on=relevant_time_name,
                how="inner",
            )
            the_lag_n_output = BestFitForecast.merge(
                lag_n_mapping, on=[o9Constants.PARTIAL_WEEK], how="inner"
            )
            the_lag_n_output = the_lag_n_output[the_lag_n_output[o9Constants.LAG] == int(LagNInput)]
            # if the_iteration is not None:
            #     the_lag_n_output[o9Constants.FORECAST_ITERATION] = the_iteration
            the_lag_n_output[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date
            the_lag_n_output.rename(
                columns={o9Constants.SYSTEM_STAT_FCST_L1: o9Constants.STAT_FCST_L1_LAGN_BACKTEST},
                inplace=True,
            )
            the_lag_n_output = the_lag_n_output[LagNFcstOutput_cols]
            lag_n_forecast_output.append(the_lag_n_output)

            if len(the_lag_mapping) == 0:
                logger.warning("the_lag_mapping is empty ...")
                continue

            # Join to get all forecast measures
            ForecastData = ForecastData.merge(
                BestFitForecast[forecast_level + [partial_week_col, system_stat_fcst_l1_col]],
                on=forecast_level + [partial_week_col],
                how="inner",
            )

            # join to get the relevant time name
            ForecastData_with_time_mapping = ForecastData.merge(
                base_time_mapping, on=partial_week_col, how="inner"
            )

            # add lag dimension column to forecast
            forecast_output_with_lag_dim = the_lag_mapping.merge(
                ForecastData_with_time_mapping,
                how="inner",
                on=relevant_time_name,
            )

            all_cols = ALL_STAT_FORECAST_COLUMNS
            cols_to_group = [x for x in all_cols if x in forecast_output_with_lag_dim.columns]
            # select the relevant columns, groupby and sum history measure
            forecast_output_with_lag_dim = (
                forecast_output_with_lag_dim.groupby(forecast_level + [relevant_time_name, lag_col])
                .sum(min_count=1)[cols_to_group + [system_stat_fcst_l1_col]]
                .reset_index()
            )

            # renaming measure names to output measures
            forecast_output_with_lag_dim.rename(
                columns=dict(
                    zip(
                        ALL_STAT_FORECAST_COLUMNS,
                        ALL_STAT_FCST_LAG_COLUMNS,
                    )
                ),
                inplace=True,
            )

            forecast_output_with_lag_dim.rename(
                columns={system_stat_fcst_l1_col: stat_fcst_l1_lag_col},
                inplace=True,
            )
            forecast_output_with_lag_dim[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date
            forecast_output_with_lag_dim[version_col] = input_version
            # forecast_output_with_lag_dim[
            #     o9Constants.FORECAST_ITERATION
            # ] = the_iteration
            for the_col in AllForecastWithLagDim_cols:
                if the_col not in forecast_output_with_lag_dim.columns:
                    forecast_output_with_lag_dim[the_col] = np.nan
            forecast_output_with_lag_dim = forecast_output_with_lag_dim[AllForecastWithLagDim_cols]

            all_forecast_with_lag_dim.append(forecast_output_with_lag_dim)
            ActualLastNBuckets[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date
            all_actual_last_n_buckets.append(ActualLastNBuckets)
            FcstNextNBuckets[o9Constants.PLANNING_CYCLE_DATE] = the_planning_cycle_date
            all_fcst_next_n_buckets.append(FcstNextNBuckets)

            logger.info(f"{df_keys}, end of cycle {the_cycle}")

        Lag1FcstOutput = concat_to_dataframe(all_lag_forecast_output)

        AllForecastWithLagDim = concat_to_dataframe(all_forecast_with_lag_dim)

        SystemBestfitAlgorithmPlanningCycle = concat_to_dataframe(
            all_systembestfitalgo_planning_cycle
        )

        LagNFcstOutput = concat_to_dataframe(lag_n_forecast_output)

        PlanningCycleAlgoStats = concat_to_dataframe(all_planning_cycle_algo_stats)
        ActualLastNBuckets = concat_to_dataframe(all_actual_last_n_buckets)
        FcstNextNBuckets = concat_to_dataframe(all_fcst_next_n_buckets)

        # Condition to check if BacktestCyclePeriods are continuous - adjacent planning cycles for stability output
        continuous_cycle_flag = np.unique(np.diff(np.array(BackTestCyclePeriods)))[0] == 1

        if all_lags_flag and continuous_cycle_flag and len(AllForecastWithLagDim) > 0:
            # filtering just the Lag backtest measure from All Forecast
            StabilityOutput = AllForecastWithLagDim[
                [grain for grain in AllForecastWithLagDim.columns if "[" in grain]
                + [stat_fcst_l1_lag_col]
            ].drop_duplicates()
            StabilityOutput.sort_values(
                by=[version_col] + forecast_level + [relevant_time_name], inplace=True
            )
            StabilityOutput.rename(columns={stat_fcst_l1_lag_col: lc_measure}, inplace=True)
            StabilityOutput[cocc_measure] = (
                StabilityOutput.groupby([version_col] + forecast_level + [relevant_time_name])[
                    lc_measure
                ]
                .diff()
                .abs()
            )
            StabilityOutput = disaggregate_data(
                source_df=StabilityOutput,
                source_grain=relevant_time_name,
                target_grain=partial_week_col,
                profile_df=StatBucketWeight.merge(
                    base_time_mapping,
                    on=o9Constants.PARTIAL_WEEK,
                    how="inner",
                ).drop(version_col, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[cocc_measure],
            )
            StabilityOutput = StabilityOutput[StabilityOutput_cols].drop_duplicates()
        else:
            logger.warning(
                f"Backtest Cycles {BackTestCyclePeriod} are not continuous, skipping Stability output ..."
            )
            StabilityOutput = pd.DataFrame(columns=StabilityOutput_cols)

        # Reasonability outputs
        if (not ActualLastNBuckets.empty) and (not FcstNextNBuckets.empty):
            ReasonabilityOutput = ActualLastNBuckets.merge(
                FcstNextNBuckets,
                on=[version_col] + forecast_level + [o9Constants.PLANNING_CYCLE_DATE],
                how="outer",
            )
            ReasonabilityOutput.rename(
                columns={
                    actual_last_n_buckets: actual_last_n_buckets_backtest,
                    fcst_next_n_buckets: fcst_next_n_buckets_backtest,
                },
                inplace=True,
            )
            ReasonabilityOutput = ReasonabilityOutput[ReasonabilityOutput_cols].drop_duplicates()
        else:
            logger.warning(
                "ActualLastNBuckets or FcstNextNBuckets is empty, skipping Reasonability output ..."
            )
            ReasonabilityOutput = pd.DataFrame(columns=ReasonabilityOutput_cols)

        if PlanningCycleAlgoStats.empty:
            PlanningCycleAlgoStats = pd.DataFrame(columns=PlanningCycleAlgoStats_cols)

        # check if output dataframes are empty
        if len(Lag1FcstOutput) == 0 and len(AllForecastWithLagDim) == 0:
            logger.warning("No backtest results generated for slice: {}...".format(df_keys))
            Lag1FcstOutput = pd.DataFrame(columns=Lag1FcstOutput_cols)
            LagNFcstOutput = pd.DataFrame(columns=LagNFcstOutput_cols)
            AllForecastWithLagDim = pd.DataFrame(columns=AllForecastWithLagDim_cols)
            SystemBestfitAlgorithmPlanningCycle = pd.DataFrame(
                columns=SystemBestfitAlgorithmPlanningCycle_cols
            )
            PlanningCycleAlgoStats = pd.DataFrame(columns=PlanningCycleAlgoStats_cols)
            StabilityOutput = pd.DataFrame(columns=StabilityOutput_cols)
            ReasonabilityOutput = pd.DataFrame(columns=ReasonabilityOutput_cols)
        if len(AllForecastWithLagDim) > 0:
            LagFcstOutput = AllForecastWithLagDim[
                [o9Constants.VERSION_NAME]
                + forecast_level
                + [
                    o9Constants.LAG,
                    o9Constants.PLANNING_CYCLE_DATE,
                    relevant_time_name,
                    stat_fcst_l1_lag_col,
                ]
            ].drop_duplicates()
            LagFcstOutput = disaggregate_data(
                source_df=LagFcstOutput,
                source_grain=relevant_time_name,
                target_grain=o9Constants.PARTIAL_WEEK,
                profile_df=StatBucketWeight.merge(
                    base_time_mapping,
                    on=o9Constants.PARTIAL_WEEK,
                    how="inner",
                ).drop(o9Constants.VERSION_NAME, axis=1),
                profile_col=o9Constants.STAT_BUCKET_WEIGHT,
                cols_to_disaggregate=[stat_fcst_l1_lag_col],
            )
            LagFcstOutput = LagFcstOutput.rename(
                columns={stat_fcst_l1_lag_col: stat_fcst_l1_lag_col_PW}
            )
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            f"Exception {e} for slice : {df_keys}, returning empty dataframe as output ..."
        )
        Lag1FcstOutput = pd.DataFrame(columns=Lag1FcstOutput_cols)
        LagNFcstOutput = pd.DataFrame(columns=LagNFcstOutput_cols)
        LagFcstOutput = pd.DataFrame(columns=LagFcstOutput_cols)
        AllForecastWithLagDim = pd.DataFrame(columns=AllForecastWithLagDim_cols)
        SystemBestfitAlgorithmPlanningCycle = pd.DataFrame(
            columns=SystemBestfitAlgorithmPlanningCycle_cols
        )
        PlanningCycleAlgoStats = pd.DataFrame(columns=PlanningCycleAlgoStats_cols)
        StabilityOutput = pd.DataFrame(columns=StabilityOutput_cols)
        ReasonabilityOutput = pd.DataFrame(columns=ReasonabilityOutput_cols)

    return (
        Lag1FcstOutput,
        LagNFcstOutput,
        LagFcstOutput,
        AllForecastWithLagDim,
        SystemBestfitAlgorithmPlanningCycle,
        PlanningCycleAlgoStats,
        StabilityOutput,
        ReasonabilityOutput,
    )
