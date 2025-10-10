import logging
import time
from typing import List, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import (
    get_last_time_period,
    get_n_time_periods,
)
from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    create_cartesian_product,
)
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data
from scipy.optimize import nnls

from helpers.DP015IdentifyBestFitModel import main as identify_best_fit_mode
from helpers.DP056OutputChecks import processIteration as output_checks
from helpers.o9Constants import o9Constants
from helpers.utils import (
    filter_for_iteration,
    get_algo_ranking,
    get_list_of_grains_from_string,
    get_rule,
)

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def calculate_planning_cycle_dates(
    validation_params_df: pd.DataFrame,
    PlanningCycleDates: pd.DataFrame,
    current_time_period_df: pd.DataFrame,
    time_dimension_df: pd.DataFrame,
    time_name_col: str,
    time_key_col: str,
    offset_period: int,
) -> List[str]:
    """
    Calculates planning cycle dates in a single, chained operation.

    Args:
        validation_params_df: DataFrame with 'Validation Period', 'Fold', 'Step Size'.
        current_time_period_df: DataFrame with one row indicating the current time.
        time_dimension_df: DataFrame mapping time names to sortable time keys.
        time_name_col: The name of the column with human-readable time (e.g., 'Month').
        time_key_col: The name of the column with a sortable date key (e.g., 'Date').

    Returns:
        A list of strings representing the calculated planning cycle dates.
    """

    def get_param(col_name: str) -> Union[int, float, None]:
        """Safely gets a single parameter from the validation DataFrame."""
        if col_name in validation_params_df.columns and not validation_params_df.empty:
            return validation_params_df[col_name].iloc[0]
        return None

    period = get_param("Validation Period")
    fold = get_param("Validation Fold")
    step = get_param("Validation Step Size")

    if not all([period, fold, step]):
        print("Warning: Missing validation parameters. Cannot generate cycles.")
        return []

    # Polars range() is equivalent to Python's built-in range()
    cycles_list = list(range(int(period), int(period) + (int(fold) * int(step)), int(step)))
    if not cycles_list:
        print("Warning: No validation cycles generated; cannot calculate dates.")
        return []

    cycles_df = pd.DataFrame({"cycle": sorted(list(set(cycles_list)))})

    latest_time_value = current_time_period_df[time_name_col].iloc[0]

    def get_date_for_cycle(cycle: int) -> Union[str, None]:
        """Safely gets the date for a single cycle number."""
        date_list = get_n_time_periods(
            latest_value=latest_time_value,
            periods=-int(cycle + offset_period),
            time_mapping=time_dimension_df,
            time_attribute={time_name_col: time_key_col},
            include_latest_value=False,
        )
        return date_list[0] if date_list else None

    planning_cycles = cycles_df["cycle"].apply(get_date_for_cycle).dropna().tolist()

    time_dimension_df["merge_key"] = time_dimension_df[o9Constants.PARTIAL_WEEK_KEY]

    planning_cycle_dates = (
        pd.merge(
            PlanningCycleDates,
            time_dimension_df,
            left_on="Planning Cycle.[PlanningCycleDateKey]",
            right_on="merge_key",
            how="inner",
        )
        .drop_duplicates(subset=["Planning Cycle.[PlanningCycleDateKey]"], keep="first")
        .sort_values(by=o9Constants.PARTIAL_WEEK_KEY)
        .loc[:, [o9Constants.PLANNING_CYCLE_DATE, time_name_col, time_key_col]]
        .loc[lambda df: df[time_name_col].isin(planning_cycles)]
        .groupby([time_name_col, time_key_col])
        .agg({o9Constants.PLANNING_CYCLE_DATE: "first"})
        .sort_values(by=time_key_col, ascending=False)
        .reset_index()[o9Constants.PLANNING_CYCLE_DATE]
        .tolist()
    )

    return planning_cycle_dates


def get_weights(
    df,
    actual_data,
    forecast_data,
    forecast_level,
    HistoryMeasure,
    partial_week_key_col,
    stat_fcst_str,
    system_ensemble_algo_list_col,
    normalized_weights_col,
    ensemble_method,
):
    # reset index
    df.reset_index(drop=True, inplace=True)

    # sort the values
    actual_data.sort_values(by=forecast_level + [partial_week_key_col], inplace=True)
    forecast_data.sort_values(by=forecast_level + [partial_week_key_col], inplace=True)

    top_n_algo_cols = [
        stat_fcst_str + x
        for x in df[system_ensemble_algo_list_col].apply(lambda x: x.split(","))[0]
    ]
    forecasts = forecast_data.merge(df[forecast_level])[top_n_algo_cols]
    actuals = actual_data.merge(df[forecast_level])[HistoryMeasure]

    weights_df = pd.DataFrame(
        columns=(forecast_level + [o9Constants.STAT_ALGORITHM, normalized_weights_col, "rank"])
    )
    if ensemble_method == "NNLS":
        weights, _ = nnls(forecasts, actuals)
        weights = np.abs(weights)  # to get positive weights
        weights = weights / weights.sum()
        weights_df = pd.DataFrame(
            [weights],
            columns=[algo.replace(stat_fcst_str, "") for algo in top_n_algo_cols],
        )

        weights_df = pd.melt(
            weights_df,
            var_name=o9Constants.STAT_ALGORITHM,
            value_name=normalized_weights_col,
        )
        weights_df = create_cartesian_product(df[forecast_level], weights_df)

        weights_df["rank"] = weights_df[normalized_weights_col].rank(ascending=False)

    elif ensemble_method == "S-OLS":
        r_squared_dict = {}
        for col in forecasts.columns:
            model = sm.OLS(actuals, forecasts[col]).fit()
            r_squared_dict[col.replace(stat_fcst_str, "")] = model.rsquared

        r_squared_dict = {
            key: abs(value) for key, value in r_squared_dict.items()
        }  # to get positive weights
        r_squared_dict = {
            key: value / sum(r_squared_dict.values()) for key, value in r_squared_dict.items()
        }

        weights_df = pd.DataFrame(
            {
                o9Constants.STAT_ALGORITHM: r_squared_dict.keys(),
                normalized_weights_col: r_squared_dict.values(),
            }
        )

        weights_df = create_cartesian_product(df[forecast_level], weights_df)

        weights_df["rank"] = weights_df[normalized_weights_col].rank(ascending=False)

    return weights_df


col_mapping = {
    "Straight Line": float,
    "Trend Violation": float,
    "Level Violation": float,
    "Seasonal Violation": float,
    "Range Violation": float,
    "COCC Violation": float,
    "Run Count": float,
    "Is Bestfit": float,
    "No Alerts": float,
    "Fcst Next N Buckets": float,
    "Validation Error": float,
    "Composite Error": float,
    "Validation Fcst": float,
    "Validation Fcst Abs Error": float,
    "Validation Method": str,
    "Run Time": float,
    "Algorithm Parameters": str,
    "Stat Fcst Ensemble": float,
    "System Bestfit Algorithm": str,
    "Validation Actual": float,
    "System Ensemble Algorithm List": str,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    TimeDimension,
    CurrentTimePeriod,
    ForecastParameters,
    ActualsAndForecastData,
    ForecastGenTimeBucket,
    AssignedAlgoList,
    Weights,
    SelectionCriteria,
    Violations,
    EnsembleParameters,
    IncludeEnsemble,
    EnsembleWeights,
    AlgoStats,
    MasterAlgoList,
    ForecastData,
    SegmentationOutput,
    StatBucketWeight,
    EnsembleOnlyStatAlgos,
    EnsembleOnlyCustomAlgos,
    Grains,
    HistoryMeasure,
    OverrideFlatLineForecasts,
    TrendVariationThreshold,
    LevelVariationThreshold,
    RangeVariationThreshold,
    SeasonalVariationThreshold,
    SeasonalVariationCountThreshold,
    ReasonabilityCycles,
    MinimumIndicePercentage,
    AbsTolerance,
    COCCVariationThreshold,
    multiprocessing_num_cores,
    df_keys,
    PlanningCycleDates=pd.DataFrame(),
    ValidationParameters=pd.DataFrame(),
    ForecastEngine=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
):
    try:
        if o9Constants.FORECAST_ITERATION in ForecastGenTimeBucket.columns:
            BestFitAlgoList = list()
            ValidationErrorList = list()
            FcstNextNBucketsList = list()
            AllEnsembleDescList = list()
            AllEnsembleFcstList = list()
            SystemEnsembleAlgorithmList = list()
            OutputAllAlgoList = list()

            for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
                logger.warning(f"--- Processing iteration {the_iteration}")

                decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

                (
                    the_best_fit_algo,
                    the_validation_error,
                    the_fcst_next_n_buckets,
                    the_all_ensemble_desc,
                    the_all_ensemble_fcst,
                    the_system_ensemble_algo_list,
                    the_output_all_algo,
                ) = decorated_func(
                    TimeDimension=TimeDimension,
                    CurrentTimePeriod=CurrentTimePeriod,
                    ForecastParameters=ForecastParameters,
                    ActualsAndForecastData=ActualsAndForecastData,
                    ForecastGenTimeBucket=ForecastGenTimeBucket,
                    AssignedAlgoList=AssignedAlgoList,
                    Weights=Weights,
                    SelectionCriteria=SelectionCriteria,
                    Violations=Violations,
                    EnsembleParameters=EnsembleParameters,
                    IncludeEnsemble=IncludeEnsemble,
                    EnsembleWeights=EnsembleWeights,
                    AlgoStats=AlgoStats,
                    MasterAlgoList=MasterAlgoList,
                    ForecastData=ForecastData,
                    SegmentationOutput=SegmentationOutput,
                    StatBucketWeight=StatBucketWeight,
                    OverrideFlatLineForecasts=OverrideFlatLineForecasts,
                    EnsembleOnlyStatAlgos=EnsembleOnlyStatAlgos,
                    EnsembleOnlyCustomAlgos=EnsembleOnlyCustomAlgos,
                    Grains=Grains,
                    HistoryMeasure=HistoryMeasure,
                    multiprocessing_num_cores=multiprocessing_num_cores,
                    TrendVariationThreshold=TrendVariationThreshold,
                    LevelVariationThreshold=LevelVariationThreshold,
                    RangeVariationThreshold=RangeVariationThreshold,
                    SeasonalVariationThreshold=SeasonalVariationThreshold,
                    SeasonalVariationCountThreshold=SeasonalVariationCountThreshold,
                    ReasonabilityCycles=ReasonabilityCycles,
                    MinimumIndicePercentage=MinimumIndicePercentage,
                    AbsTolerance=AbsTolerance,
                    COCCVariationThreshold=COCCVariationThreshold,
                    the_iteration=the_iteration,
                    df_keys=df_keys,
                    PlanningCycleDates=PlanningCycleDates,
                    ValidationParameters=ValidationParameters,
                    ForecastEngine=ForecastEngine,
                    SellOutOffset=SellOutOffset,
                )
                BestFitAlgoList.append(the_best_fit_algo)
                ValidationErrorList.append(the_validation_error)
                FcstNextNBucketsList.append(the_fcst_next_n_buckets)
                AllEnsembleDescList.append(the_all_ensemble_desc)
                AllEnsembleFcstList.append(the_all_ensemble_fcst)
                SystemEnsembleAlgorithmList.append(the_system_ensemble_algo_list)
                OutputAllAlgoList.append(the_output_all_algo)

            BestFitAlgo = concat_to_dataframe(BestFitAlgoList)
            ValidationError = concat_to_dataframe(ValidationErrorList)
            FcstNextNBuckets = concat_to_dataframe(FcstNextNBucketsList)
            AllEnsembleDesc = concat_to_dataframe(AllEnsembleDescList)
            AllEnsembleFcst = concat_to_dataframe(AllEnsembleFcstList)
            SystemEnsembleAlgorithmList = concat_to_dataframe(SystemEnsembleAlgorithmList)
            OutputAllAlgo = concat_to_dataframe(OutputAllAlgoList)

        else:
            (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            ) = processIteration(
                TimeDimension=TimeDimension,
                CurrentTimePeriod=CurrentTimePeriod,
                ForecastParameters=ForecastParameters,
                ActualsAndForecastData=ActualsAndForecastData,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                AssignedAlgoList=AssignedAlgoList,
                Weights=Weights,
                SelectionCriteria=SelectionCriteria,
                Violations=Violations,
                EnsembleParameters=EnsembleParameters,
                IncludeEnsemble=IncludeEnsemble,
                EnsembleWeights=EnsembleWeights,
                AlgoStats=AlgoStats,
                MasterAlgoList=MasterAlgoList,
                ForecastData=ForecastData,
                SegmentationOutput=SegmentationOutput,
                StatBucketWeight=StatBucketWeight,
                OverrideFlatLineForecasts=OverrideFlatLineForecasts,
                EnsembleOnlyStatAlgos=EnsembleOnlyStatAlgos,
                EnsembleOnlyCustomAlgos=EnsembleOnlyCustomAlgos,
                Grains=Grains,
                HistoryMeasure=HistoryMeasure,
                multiprocessing_num_cores=multiprocessing_num_cores,
                TrendVariationThreshold=TrendVariationThreshold,
                LevelVariationThreshold=LevelVariationThreshold,
                RangeVariationThreshold=RangeVariationThreshold,
                SeasonalVariationThreshold=SeasonalVariationThreshold,
                SeasonalVariationCountThreshold=SeasonalVariationCountThreshold,
                ReasonabilityCycles=ReasonabilityCycles,
                MinimumIndicePercentage=MinimumIndicePercentage,
                AbsTolerance=AbsTolerance,
                COCCVariationThreshold=COCCVariationThreshold,
                the_iteration=None,
                df_keys=df_keys,
                PlanningCycleDates=PlanningCycleDates,
                ValidationParameters=ValidationParameters,
                ForecastEngine=ForecastEngine,
                SellOutOffset=SellOutOffset,
            )

    except Exception as e:
        logger.exception(e)
        BestFitAlgo, ValidationError = None, None
        FcstNextNBuckets, AllEnsembleDesc = None, None
        AllEnsembleFcst, SystemEnsembleAlgorithmList, OutputAllAlgo = (
            None,
            None,
            None,
        )

    return (
        BestFitAlgo,
        ValidationError,
        FcstNextNBuckets,
        AllEnsembleDesc,
        AllEnsembleFcst,
        SystemEnsembleAlgorithmList,
        OutputAllAlgo,
    )


def processIteration(
    TimeDimension,
    CurrentTimePeriod,
    ForecastParameters,
    ActualsAndForecastData,
    ForecastGenTimeBucket,
    AssignedAlgoList,
    Weights,
    SelectionCriteria,
    Violations,
    EnsembleParameters,
    IncludeEnsemble,
    EnsembleWeights,
    AlgoStats,
    MasterAlgoList,
    ForecastData,
    SegmentationOutput,
    StatBucketWeight,
    EnsembleOnlyStatAlgos,
    EnsembleOnlyCustomAlgos,
    Grains,
    HistoryMeasure,
    OverrideFlatLineForecasts,
    TrendVariationThreshold,
    LevelVariationThreshold,
    RangeVariationThreshold,
    SeasonalVariationThreshold,
    SeasonalVariationCountThreshold,
    ReasonabilityCycles,
    MinimumIndicePercentage,
    AbsTolerance,
    COCCVariationThreshold,
    multiprocessing_num_cores,
    df_keys,
    the_iteration=None,
    ValidationParameters=pd.DataFrame(),
    PlanningCycleDates=pd.DataFrame(),
    ForecastEngine=pd.DataFrame(),
    SellOutOffset=pd.DataFrame(),
):
    plugin_name = "DP069EnsembleFcst"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    partial_week_col = o9Constants.PARTIAL_WEEK
    week_col = o9Constants.WEEK
    month_col = o9Constants.MONTH
    planning_month_col = o9Constants.PLANNING_MONTH
    quarter_col = o9Constants.QUARTER
    planning_quarter_col = o9Constants.PLANNING_QUARTER

    partial_week_key_col = o9Constants.PARTIAL_WEEK_KEY
    week_key_col = o9Constants.WEEK_KEY
    month_key_col = o9Constants.MONTH_KEY
    planning_month_key_col = o9Constants.PLANNING_MONTH_KEY
    quarter_key_col = o9Constants.QUARTER_KEY
    planning_quarter_key_col = o9Constants.PLANNING_QUARTER_KEY

    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    assigned_algo_list_col = "Assigned Algorithm List"
    assigned_rule_col = "Assigned Rule"
    history_period_col = "History Period"
    forecast_period_col = "Forecast Period"
    validation_period_col = "Validation Period"
    bestfit_method_col = "Bestfit Method"
    error_metric_col = "Error Metric"
    history_time_buckets_col = "History Time Buckets"
    stat_fcst_str = "Stat Fcst "

    ensemble_method_col = "Ensemble Method"
    ensemble_top_n_value_col = "Ensemble Top N Value"
    error_col = "Error"
    inv_error_col = "Inv Error"
    relevant_algo_col = "Relevant Algos"
    normalized_weights_col = "Normalized Weights"
    stat_bucket_weight_col = "Stat Bucket Weight"
    include_ensemble_col = "Include Ensemble"
    forecast_strategy_col = "Forecast Strategy"
    planner_ensemble_wt_col = "Planner Ensemble Weight"
    planner_stat_fcst_l1_col = "Planner Stat Fcst L1"
    stat_fcst_l1_col = "Stat Fcst L1"
    stat_fcst_l1_lc_col = "Stat Fcst L1 LC"
    round_decimals = 3

    # validation error output cols
    validation_error_col = o9Constants.VALIDATION_ERROR
    composite_error_col = o9Constants.COMPOSITE_ERROR
    validation_period_agg_actual_col = "Validation Actual"
    validation_period_agg_forecast_col = "Validation Fcst"
    validation_period_fcst_abs_error_col = "Validation Fcst Abs Error"

    # bestfit output col
    best_fit_algo_col = o9Constants.SYSTEM_BESTFIT_ALGORITHM

    # all algo output cols
    straight_line_col = "Straight Line"
    trend_col = "Trend Violation"
    seasonal_col = "Seasonal Violation"
    level_col = "Level Violation"
    range_col = "Range Violation"
    cocc_col = "COCC Violation"
    run_count_col = "Run Count"
    is_bestfit_col = "Is Bestfit"
    no_alerts_col = "No Alerts"
    sell_out_offset_col = "Offset Period"

    # fcst next n buckets output col
    fcst_next_n_buckets_col = "Fcst Next N Buckets"

    # ensemble fcst output col
    stat_fcst_ensemble_col = "Stat Fcst Ensemble"

    # system ensemble algo list output col
    system_ensemble_algo_list_col = "System Ensemble Algorithm List"

    # collect forecast column names
    ALL_STAT_FORECAST_COLS = [x for x in ActualsAndForecastData.columns if stat_fcst_str in x]

    # all stat algos
    logger.info("Getting all stat algos and custom algos ...")
    stat_algo_list = get_list_of_grains_from_string(
        input=MasterAlgoList[assigned_algo_list_col].values[0]
    )
    stat_algo_list = [stat_fcst_str + x for x in stat_algo_list]
    custom_algo_list = [
        x.replace(stat_fcst_str, "") for x in ALL_STAT_FORECAST_COLS if x not in stat_algo_list
    ] + ["Ensemble"]
    stat_algo_list = [x.replace(stat_fcst_str, "") for x in stat_algo_list]

    logger.info("Extracting forecast level ...")
    forecast_level = get_list_of_grains_from_string(input=Grains)

    BestFitAlgo_cols = [o9Constants.VERSION_NAME] + forecast_level + [best_fit_algo_col]
    BestFitAlgo = pd.DataFrame(columns=BestFitAlgo_cols)

    ValidationError_cols = (
        [o9Constants.VERSION_NAME]
        + forecast_level
        + [
            o9Constants.STAT_ALGORITHM,
            o9Constants.STAT_RULE,
            validation_error_col,
            composite_error_col,
            validation_period_agg_actual_col,
            validation_period_agg_forecast_col,
            validation_period_fcst_abs_error_col,
        ]
    )
    ValidationError = pd.DataFrame(columns=ValidationError_cols)

    OutputAllAlgo_cols = (
        [o9Constants.VERSION_NAME]
        + forecast_level
        + [
            o9Constants.STAT_ALGORITHM,
            o9Constants.STAT_RULE,
            straight_line_col,
            trend_col,
            level_col,
            seasonal_col,
            range_col,
            cocc_col,
            run_count_col,
            is_bestfit_col,
            no_alerts_col,
            o9Constants.VALIDATION_METHOD,
        ]
    )
    OutputAllAlgo = pd.DataFrame(columns=OutputAllAlgo_cols)

    FcstNextNBuckets_cols = (
        [o9Constants.VERSION_NAME]
        + forecast_level
        + [
            o9Constants.STAT_ALGORITHM,
            o9Constants.STAT_RULE,
            fcst_next_n_buckets_col,
        ]
    )
    FcstNextNBuckets = pd.DataFrame(columns=FcstNextNBuckets_cols)

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
    AllEnsembleDesc = pd.DataFrame(columns=cols_required_in_all_ensemble_desc_df)

    cols_required_in_all_ensemble_fcst_df = (
        [o9Constants.VERSION_NAME]
        + forecast_level
        + [
            partial_week_col,
            stat_fcst_ensemble_col,
        ]
    )
    AllEnsembleFcst = pd.DataFrame(columns=cols_required_in_all_ensemble_fcst_df)

    cols_required_in_system_ensemble_algo_list = (
        [o9Constants.VERSION_NAME]
        + forecast_level
        + [
            system_ensemble_algo_list_col,
        ]
    )
    SystemEnsembleAlgorithmList = pd.DataFrame(columns=cols_required_in_system_ensemble_algo_list)

    try:
        if AlgoStats.empty:
            logger.warning(f"AlgoStats is empty for slice {df_keys}. Returning empty dataframe...")
            return (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            )

        # adding forecast iteration to use it later in identify bestfit and output checks plugin
        # if the_iteration is not None:
        #     AlgoStats[o9Constants.FORECAST_ITERATION] = the_iteration
        #     ActualsAndForecastData[
        #         o9Constants.FORECAST_ITERATION
        #     ] = the_iteration
        #     ForecastGenTimeBucket[
        #         o9Constants.FORECAST_ITERATION
        #     ] = the_iteration
        #     ForecastData[o9Constants.FORECAST_ITERATION] = the_iteration
        #     ForecastParameters[o9Constants.FORECAST_ITERATION] = the_iteration

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
        req_cols = (
            [
                o9Constants.VERSION_NAME,
                o9Constants.STAT_RULE,
                o9Constants.STAT_ALGORITHM,
            ]
            + forecast_level
            + [validation_error_col, composite_error_col]
        )
        ensemble_weights = AlgoStats[req_cols]

        # check rules which have true values and consider only those if any
        if not IncludeEnsemble.empty and IncludeEnsemble[include_ensemble_col].any():
            rules_list = IncludeEnsemble[IncludeEnsemble[include_ensemble_col]][
                o9Constants.STAT_RULE
            ].to_list()
            ensemble_weights = ensemble_weights[
                ensemble_weights[o9Constants.STAT_RULE].isin(rules_list)
            ]

        if ActualsAndForecastData.empty:
            logger.warning(f"ActualsAndForecastData is empty for slice {df_keys}")
            return (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            )

        req_cols = (
            [
                o9Constants.VERSION_NAME,
            ]
            + forecast_level
            + [partial_week_col, o9Constants.ACTUAL_CLEANSED, o9Constants.STAT_ACTUAL]
        )
        Actuals = ActualsAndForecastData[req_cols]

        if Actuals.empty:
            logger.warning(f"No data for measure {o9Constants.ACTUAL_CLEANSED} for slice {df_keys}")
            logger.warning("Returning empty dataframe ...")
            return (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            )

        if ForecastData.empty or SegmentationOutput.empty or SelectionCriteria.empty:
            logger.warning(
                f"ForecastData/SegmentationOutput/SelectionCriteria is empty for slice {df_keys}"
            )
            logger.warning("Returning empty dataframe ...")
            return (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            )

        if TimeDimension.empty or ForecastParameters.empty or EnsembleParameters.empty:
            logger.warning(
                f"TimeDimension/ForecastParameters/EnsembleParameters is empty for slice {df_keys}"
            )
            logger.warning("Returning empty dataframe ...")
            return (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            )

        req_cols = (
            [
                o9Constants.VERSION_NAME,
            ]
            + forecast_level
            + [
                partial_week_col,
                planner_stat_fcst_l1_col,
                stat_fcst_l1_col,
                stat_fcst_l1_lc_col,
            ]
        )
        ForecastData = ForecastData[req_cols]

        req_cols = (
            [o9Constants.VERSION_NAME]
            + forecast_level
            + [partial_week_col, HistoryMeasure]
            + ALL_STAT_FORECAST_COLS
        )
        ActualsAndForecastData = ActualsAndForecastData[req_cols]

        input_version = ActualsAndForecastData[o9Constants.VERSION_NAME].iloc[0]

        if AssignedAlgoList.empty:
            logger.warning(f"AssignedAlgoList is empty for slice {df_keys}")
            logger.warning("Assigning default values for all stat intersections")
            AssignedAlgoList = ActualsAndForecastData[
                [o9Constants.VERSION_NAME] + forecast_level
            ].drop_duplicates()
            AssignedAlgoList[assigned_rule_col] = "Custom"
            AssignedAlgoList[assigned_algo_list_col] = ",".join(custom_algo_list)
            AssignedAlgoList[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST] = ""
            AssignedAlgoList[o9Constants.PLANNER_BESTFIT_ALGORITHM] = ""

        else:
            AssignedAlgoList[assigned_algo_list_col] = (
                AssignedAlgoList[assigned_algo_list_col] + "," + ",".join(custom_algo_list)
            )

        req_cols = [
            o9Constants.VERSION_NAME,
            forecast_period_col,
            history_period_col,
        ]
        ForecastSetupConfiguration = ForecastParameters[req_cols]

        # if ~EnsembleOnlyStatAlgos and ~EnsembleOnlyCustomAlgos:
        #     logger.warning(
        #         "EnsembleOnlyStatAlgos and EnsembleOnlyStatAlgos both can't be false ..."
        #     )
        #     logger.warning("Making EnsembleOnlyStatAlgos as True ...")
        #     EnsembleOnlyStatAlgos = True

        if EnsembleOnlyStatAlgos and EnsembleOnlyCustomAlgos:
            logger.warning("EnsembleOnlyStatAlgos and EnsembleOnlyStatAlgos both can't be true ...")
            logger.warning("Considering only EnsembleOnlyStatAlgos as True ...")
            EnsembleOnlyCustomAlgos = False

        if EnsembleOnlyStatAlgos:
            ensemble_weights = ensemble_weights[
                ensemble_weights[o9Constants.STAT_ALGORITHM].isin(stat_algo_list)
            ]
        elif EnsembleOnlyCustomAlgos:
            ensemble_weights = ensemble_weights[
                ensemble_weights[o9Constants.STAT_ALGORITHM].isin(custom_algo_list)
            ]

        req_cols = [
            o9Constants.VERSION_NAME,
            history_period_col,
            validation_period_col,
            bestfit_method_col,
            error_metric_col,
            history_time_buckets_col,
            forecast_period_col,
            forecast_strategy_col,
        ]
        ForecastParameters = ForecastParameters[req_cols]

        validation_periods = int(ForecastParameters[validation_period_col].iloc[0])
        forecast_periods = int(ForecastParameters[forecast_period_col].iloc[0])
        forecast_strategy = ForecastParameters[forecast_strategy_col].iloc[0]
        top_n_algos_number = EnsembleParameters[ensemble_top_n_value_col].iloc[0]
        if np.isnan(top_n_algos_number):
            logger.warning("No value is there for top n algos. Setting default value as 3 ...")
            top_n_algos_number = 3

        logger.debug(f"validation_periods : {validation_periods}")
        logger.debug(f"forecast_periods : {forecast_periods}")
        logger.debug(f"forecast_strategy : {forecast_strategy}")
        logger.debug(f"top_n_algos_number : {top_n_algos_number}")

        logger.info("Checking if given ensemble method is valid or not ...")
        ensemble_method = EnsembleParameters[ensemble_method_col].iloc[0]
        valid_methods = ["NNLS", "Error Weighted", "S-OLS", "Simple Avg"]
        if ensemble_method not in valid_methods:
            logger.warning(
                f"Check ensemble method. {valid_methods} ensemble methods are acceptable and given method is {ensemble_method} ..."
            )
            logger.warning(f"Will return empty dataframe for slice {df_keys}...")
            return (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            )

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[fcst_gen_time_bucket_col].unique()[0]
        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        if fcst_gen_time_bucket == "Week":
            frequency = "Weekly"
            relevant_time_cols = [partial_week_col, week_col, week_key_col]
            relevant_time_name = week_col
            relevant_time_key = week_key_col
        elif fcst_gen_time_bucket == "Month":
            frequency = "Monthly"
            relevant_time_cols = [partial_week_col, month_col, month_key_col]
            relevant_time_name = month_col
            relevant_time_key = month_key_col
        elif fcst_gen_time_bucket == "Planning Month":
            frequency = "Monthly"
            relevant_time_cols = [
                partial_week_col,
                planning_month_col,
                planning_month_key_col,
            ]
            relevant_time_name = planning_month_col
            relevant_time_key = planning_month_key_col
        elif fcst_gen_time_bucket == "Planning Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                partial_week_col,
                planning_quarter_col,
                planning_quarter_key_col,
            ]
            relevant_time_name = planning_quarter_col
            relevant_time_key = planning_quarter_key_col
        elif fcst_gen_time_bucket == "Quarter":
            frequency = "Quarterly"
            relevant_time_cols = [
                partial_week_col,
                quarter_col,
                quarter_key_col,
            ]
            relevant_time_name = quarter_col
            relevant_time_key = quarter_key_col
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            )

        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # retain time mapping with partial week
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()

        # Join Actuals with time mapping
        relevant_actual_forecast_data = ActualsAndForecastData.merge(
            base_time_mapping, on=partial_week_col, how="inner"
        )
        if relevant_actual_forecast_data.empty:
            logger.warning(
                f"No relevant data found. Will return empty dataframe for slice {df_keys} ..."
            )
            return (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            )

        list_of_measures_to_agg = [HistoryMeasure] + ALL_STAT_FORECAST_COLS

        # select the relevant columns, groupby and sum history measure
        # Change the datatype of the HistoryMeasure col
        relevant_actual_forecast_data[HistoryMeasure] = relevant_actual_forecast_data[
            HistoryMeasure
        ].astype("float64")

        # using it later to calculate weights
        relevant_data = relevant_actual_forecast_data.copy()
        relevant_data.fillna(0, inplace=True)

        logger.debug(relevant_actual_forecast_data[HistoryMeasure].dtypes)

        relevant_actual_forecast_data = (
            relevant_actual_forecast_data.groupby(forecast_level + [relevant_time_name])
            .sum(min_count=1)[list_of_measures_to_agg]
            .reset_index()
        )

        logger.debug(
            f"relevant_actual_forecast_data cols after groupby : {relevant_actual_forecast_data.columns}"
        )
        logger.debug(
            f"relevant_actual_forecast_data dtypes : {relevant_actual_forecast_data.dtypes}"
        )

        # get last time period
        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )

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

        # get the validation dates
        validation_period_dates = get_n_time_periods(
            latest_time_name,
            -validation_periods,
            relevant_time_mapping,
            time_attribute_dict,
        )

        # get the forecast dates
        forecast_period_dates = get_n_time_periods(
            latest_time_name,
            forecast_periods + offset_periods,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=False,
        )

        relevant_dates = pd.DataFrame(
            {relevant_time_name: validation_period_dates + forecast_period_dates}
        )

        # filter relevant dates
        relevant_actual_forecast_data = relevant_actual_forecast_data.merge(relevant_dates)

        logger.info("getting ensemble weights ...")
        # adding this check if in case ensemble results are not nulled out
        # to avoid having ensemble in top n algos
        EnsembleWeights = EnsembleWeights[EnsembleWeights[o9Constants.STAT_ALGORITHM] != "Ensemble"]
        EnsembleWeights.dropna(subset=[planner_ensemble_wt_col], inplace=True)
        if not EnsembleWeights.empty:
            logger.info("calculating weights based on planner weights ...")
            # group by the forecast_level and aggregate to find non-null algos
            logger.info("getting non null algo for each intersection ...")
            non_null_algos = relevant_actual_forecast_data.groupby(forecast_level)[
                ALL_STAT_FORECAST_COLS
            ].apply(lambda x: x.notnull().any())

            # get a comma separated non null algos
            non_null_algos = non_null_algos.apply(
                lambda row: ",".join([col.replace(stat_fcst_str, "") for col in row[row].index]),
                axis=1,
            ).reset_index(name=relevant_algo_col)

            relevant_algos = pd.DataFrame(non_null_algos)

            EnsembleWeights = EnsembleWeights.merge(relevant_algos)
            EnsembleWeights[relevant_algo_col] = EnsembleWeights[relevant_algo_col].apply(
                lambda x: list(x.split(","))
            )

            EnsembleWeights = EnsembleWeights.explode(relevant_algo_col)
            EnsembleWeights = EnsembleWeights[
                EnsembleWeights[relevant_algo_col] == EnsembleWeights[o9Constants.STAT_ALGORITHM]
            ]

            if EnsembleWeights.empty:
                logger.warning("no relevant algo left after dropping algos with no forecast ...")
                logger.warning(f"Returning empty dataframe for slice {df_keys} ...")
                return (
                    BestFitAlgo,
                    ValidationError,
                    FcstNextNBuckets,
                    AllEnsembleDesc,
                    AllEnsembleFcst,
                    SystemEnsembleAlgorithmList,
                    OutputAllAlgo,
                )

            # filter out relevant columns
            EnsembleWeights = EnsembleWeights[
                forecast_level + [o9Constants.STAT_ALGORITHM, planner_ensemble_wt_col]
            ].drop_duplicates()

            # normalizing the planner given weights
            logger.info("normalizing the weights ...")
            weight_calc_start_time = time.time()
            EnsembleWeights[normalized_weights_col] = EnsembleWeights.groupby(
                forecast_level, observed=True
            )[planner_ensemble_wt_col].transform("sum")
            EnsembleWeights[normalized_weights_col] = (
                EnsembleWeights[planner_ensemble_wt_col] / EnsembleWeights[normalized_weights_col]
            )
            weight_calc_end_time = time.time()

            ensemble_weights = EnsembleWeights[
                forecast_level + [o9Constants.STAT_ALGORITHM, normalized_weights_col]
            ]

            # convert the stat algo column to categorical so that sort order is preserved
            # custom should be prioritized over Stat
            ensemble_weights[o9Constants.STAT_ALGORITHM] = pd.Categorical(
                ensemble_weights[o9Constants.STAT_ALGORITHM],
                categories=(
                    custom_algo_list + [x for x in get_algo_ranking() if x not in custom_algo_list]
                ),
                ordered=True,
            )
            # sort the dataframe by stat algo col
            ensemble_weights.sort_values(
                forecast_level + [o9Constants.STAT_ALGORITHM], inplace=True
            )

            # revert categorical dtype to str
            ensemble_weights[o9Constants.STAT_ALGORITHM] = ensemble_weights[
                o9Constants.STAT_ALGORITHM
            ].astype("str")

            ensemble_weights["rank"] = ensemble_weights.groupby(forecast_level)[
                normalized_weights_col
            ].rank(method="first", ascending=False)

        else:
            logger.info("calculating weights based on selected ensemble method ...")

            if (ensemble_weights[composite_error_col].isna().all()) & (
                ensemble_weights[validation_error_col].isna().all()
            ):
                logger.warning("Vaidation and composite both can't be null ...")
                logger.warning("Check the input. Returning empty dataframe ...")
                return (
                    BestFitAlgo,
                    ValidationError,
                    FcstNextNBuckets,
                    AllEnsembleDesc,
                    AllEnsembleFcst,
                    SystemEnsembleAlgorithmList,
                    OutputAllAlgo,
                )
            ensemble_weights = ensemble_weights[
                ~(
                    ensemble_weights[composite_error_col].isna()
                    & ensemble_weights[validation_error_col].isna()
                )
            ]
            # consider validation error if composite error is not present
            ensemble_weights[error_col] = (
                ensemble_weights[composite_error_col]
                if composite_error_col in ensemble_weights.columns
                else ensemble_weights[validation_error_col]
            )
            ensemble_weights[inv_error_col] = np.where(
                ensemble_weights[error_col] != 0,
                1 / ensemble_weights[error_col],
                1,
            )

            # check if any values are not in master list
            missing_algos = [
                x
                for x in ensemble_weights[o9Constants.STAT_ALGORITHM].unique()
                if x not in list(set(custom_algo_list).union(set(get_algo_ranking())))
            ]

            # Removing Ensemble if present
            ensemble_weights = ensemble_weights[
                ensemble_weights[o9Constants.STAT_ALGORITHM] != "Ensemble"
            ]

            if missing_algos:
                logger.warning(
                    f"Please add {missing_algos} to the get_algo_ranking() function and run again"
                )

                # filter out to avoid downstream issues
                ensemble_weights = ensemble_weights[
                    ~ensemble_weights[o9Constants.STAT_ALGORITHM].isin(missing_algos)
                ]

            # convert the stat algo column to categorical so that sort order is preserved
            # custom should be prioritized over Stat
            ensemble_weights[o9Constants.STAT_ALGORITHM] = pd.Categorical(
                ensemble_weights[o9Constants.STAT_ALGORITHM],
                categories=(
                    custom_algo_list + [x for x in get_algo_ranking() if x not in custom_algo_list]
                ),
                ordered=True,
            )
            # sort the dataframe by stat algo col
            ensemble_weights.sort_values(
                forecast_level + [o9Constants.STAT_ALGORITHM], inplace=True
            )

            # revert categorical dtype to str
            ensemble_weights[o9Constants.STAT_ALGORITHM] = ensemble_weights[
                o9Constants.STAT_ALGORITHM
            ].astype("str")

            # Find the rank based on the minimum error for each group
            ensemble_weights["rank"] = ensemble_weights.groupby(forecast_level)[error_col].rank(
                method="first"
            )

            ensemble_weights = ensemble_weights[ensemble_weights["rank"] <= top_n_algos_number]

            top_n_algos_list_df = (
                ensemble_weights.groupby(forecast_level, observed=True)
                .apply(lambda x: ",".join(x[o9Constants.STAT_ALGORITHM]))
                .reset_index(name=system_ensemble_algo_list_col)
            )

            relevant_data = relevant_data[
                relevant_data[relevant_time_name].isin(validation_period_dates)
            ]
            relevant_data = relevant_data.merge(
                TimeDimension[[partial_week_col, partial_week_key_col]].drop_duplicates()
            )

            relevant_actuals = relevant_data[
                forecast_level + [partial_week_col, partial_week_key_col, HistoryMeasure]
            ]

            weight_calc_start_time = time.time()
            if ensemble_method == "Error Weighted":
                ensemble_weights["sum"] = ensemble_weights.groupby(forecast_level)[
                    inv_error_col
                ].transform("sum")
                ensemble_weights[normalized_weights_col] = (
                    ensemble_weights[inv_error_col] / ensemble_weights["sum"]
                )
                ensemble_weights = ensemble_weights[
                    forecast_level
                    + [
                        o9Constants.STAT_ALGORITHM,
                        normalized_weights_col,
                        "rank",
                    ]
                ]

            elif ensemble_method == "Simple Avg":
                ensemble_weights[normalized_weights_col] = 1 / top_n_algos_number
                ensemble_weights = ensemble_weights[
                    forecast_level
                    + [
                        o9Constants.STAT_ALGORITHM,
                        normalized_weights_col,
                        "rank",
                    ]
                ]

            else:
                weight_list = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
                    delayed(get_weights)(
                        df=group,
                        actual_data=relevant_actuals,
                        forecast_data=relevant_data,
                        forecast_level=forecast_level,
                        HistoryMeasure=HistoryMeasure,
                        partial_week_key_col=partial_week_key_col,
                        stat_fcst_str=stat_fcst_str,
                        system_ensemble_algo_list_col=system_ensemble_algo_list_col,
                        normalized_weights_col=normalized_weights_col,
                        ensemble_method=ensemble_method,
                    )
                    for name, group in top_n_algos_list_df.groupby(forecast_level)
                )
                ensemble_weights = concat_to_dataframe(weight_list)

            weight_calc_end_time = time.time()

        # sort values based on forecast level and rank
        logger.info("sorting ensemble weights based on forecast level and rank ...")
        ensemble_weights.sort_values(
            by=forecast_level + ["rank"],
            axis=0,
            inplace=True,
            ignore_index=True,
        )

        # getting ensemble algo and corresponding weights
        # DES = 0.2 | Theta = 0.5 | STLF = 0.3
        all_ensemble_desc_df = (
            ensemble_weights.groupby(forecast_level, observed=True)
            .apply(
                lambda x: " | ".join(
                    x[o9Constants.STAT_ALGORITHM]
                    + " = "
                    + x[normalized_weights_col].round(round_decimals).astype(str)
                )
            )
            .reset_index(name=o9Constants.ALGORITHM_PARAMETERS)
        )

        system_ensemble_algo_list = (
            ensemble_weights.groupby(forecast_level, observed=True)
            .apply(lambda x: ",".join(x[o9Constants.STAT_ALGORITHM]))
            .reset_index(name=system_ensemble_algo_list_col)
        )

        system_ensemble_algo_list.insert(0, o9Constants.VERSION_NAME, input_version)

        # adding stat algo column
        all_ensemble_desc_df[o9Constants.STAT_ALGORITHM] = "Ensemble"

        # adding stat fcst for all algos for merging purpose
        ensemble_weights[o9Constants.STAT_ALGORITHM] = (
            stat_fcst_str + ensemble_weights[o9Constants.STAT_ALGORITHM]
        )
        ensemble_weights = ensemble_weights[
            forecast_level + [o9Constants.STAT_ALGORITHM, normalized_weights_col]
        ]

        relevant_actual_forecast_data = relevant_actual_forecast_data.merge(
            ensemble_weights[forecast_level].drop_duplicates()
        )

        # Melt the DataFrame to reshape it for pivoting
        ActualsAndForecastData_pivot = relevant_actual_forecast_data.melt(
            id_vars=forecast_level + [relevant_time_name],
            value_vars=ALL_STAT_FORECAST_COLS,
            var_name=o9Constants.STAT_ALGORITHM,
            value_name="Fcst Value",
        )

        # Pivot the DataFrame
        ActualsAndForecastData_pivot = ActualsAndForecastData_pivot.pivot_table(
            index=forecast_level + [o9Constants.STAT_ALGORITHM],
            columns=relevant_time_name,
            values="Fcst Value",
        ).reset_index()

        fcst_start_time = time.time()

        ActualsAndForecastData_pivot = ActualsAndForecastData_pivot.merge(ensemble_weights)
        cols_to_consider = [
            x
            for x in ActualsAndForecastData_pivot.columns
            if x not in forecast_level + [o9Constants.STAT_ALGORITHM, normalized_weights_col]
        ]
        ActualsAndForecastData_pivot[cols_to_consider] = ActualsAndForecastData_pivot[
            cols_to_consider
        ].multiply(ActualsAndForecastData_pivot[normalized_weights_col], axis=0)

        ActualsAndForecastData_pivot.drop(columns=[normalized_weights_col], inplace=True)
        ActualsAndForecastData_pivot = pd.melt(
            ActualsAndForecastData_pivot,
            id_vars=forecast_level + [o9Constants.STAT_ALGORITHM],
            var_name=relevant_time_name,
            value_name="Fcst Value",
        )
        ActualsAndForecastData_pivot = ActualsAndForecastData_pivot.pivot(
            index=forecast_level + [relevant_time_name],
            columns=o9Constants.STAT_ALGORITHM,
            values="Fcst Value",
        ).reset_index()

        existing_columns = [
            col for col in ALL_STAT_FORECAST_COLS if col in ActualsAndForecastData_pivot.columns
        ]
        ActualsAndForecastData_pivot[stat_fcst_ensemble_col] = ActualsAndForecastData_pivot[
            existing_columns
        ].sum(axis=1, skipna=True)

        fcst_end_time = time.time()

        # total time is time taken to calculate weights and generating forecast
        total_time_taken = (fcst_end_time - fcst_start_time) + (
            weight_calc_end_time - weight_calc_start_time
        )
        # assuming each intersection took same time
        time_taken_for_an_intersection = total_time_taken / len(
            ActualsAndForecastData_pivot[forecast_level].drop_duplicates()
        )

        all_ensemble_desc_df[o9Constants.RUN_TIME] = time_taken_for_an_intersection
        all_ensemble_desc_df.insert(0, o9Constants.VERSION_NAME, input_version)

        ActualsAndForecastData_pivot.insert(0, o9Constants.VERSION_NAME, input_version)

        cols_required = (
            [o9Constants.VERSION_NAME]
            + forecast_level
            + [relevant_time_name, stat_fcst_ensemble_col]
        )
        ActualsAndForecastData_pivot = ActualsAndForecastData_pivot[cols_required]

        # rounding off
        ActualsAndForecastData_pivot = ActualsAndForecastData_pivot.round(round_decimals)

        # get statbucket weights at the desired level
        StatBucketWeight = StatBucketWeight.merge(
            base_time_mapping, on=partial_week_col, how="inner"
        )

        # perform disaggregation
        all_ensemble_fcst_df = disaggregate_data(
            source_df=ActualsAndForecastData_pivot,
            source_grain=relevant_time_name,
            target_grain=partial_week_col,
            profile_df=StatBucketWeight.drop(o9Constants.VERSION_NAME, axis=1),
            profile_col=stat_bucket_weight_col,
            cols_to_disaggregate=[stat_fcst_ensemble_col],
        )

        all_ensemble_fcst_identify = all_ensemble_fcst_df.copy()

        # Caclulation of Valid Planning Cycles
        planning_cycle_dates = calculate_planning_cycle_dates(
            validation_params_df=ValidationParameters,
            PlanningCycleDates=PlanningCycleDates,
            current_time_period_df=CurrentTimePeriod,
            time_dimension_df=TimeDimension,
            time_name_col=relevant_time_name,
            time_key_col=relevant_time_key,
            offset_period=offset_periods,
        )

        # getting latest planning cycle date
        latest_planning_cycle_date = planning_cycle_dates[0]

        all_ensemble_fcst_identify[o9Constants.PLANNING_CYCLE_DATE] = latest_planning_cycle_date

        all_ensemble_fcst_identify["Stat Fcst Ensemble Planning Cycle"] = (
            all_ensemble_fcst_identify["Stat Fcst Ensemble"]
        )

        all_ensemble_fcst_identify["Stat Fcst CML Planning Cycle"] = np.nan

        # need to calculate errors for ensemble and bestfit between all algos
        AllForecast = all_ensemble_fcst_df.merge(ActualsAndForecastData)
        for col in custom_algo_list:
            if stat_fcst_str + col not in AllForecast.columns:
                AllForecast[stat_fcst_str + col] = np.nan

        # Identify best fit model
        ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION] = the_iteration
        SelectionCriteria[o9Constants.FORECAST_ITERATION] = the_iteration
        ForecastParameters[o9Constants.FORECAST_ITERATION] = the_iteration
        all_ensemble_fcst_identify[o9Constants.FORECAST_ITERATION] = the_iteration

        all_ensemble_fcst_identify = all_ensemble_fcst_identify.drop(columns="Stat Fcst Ensemble")

        Actuals_identify = AllForecast[
            [o9Constants.VERSION_NAME] + forecast_level + [o9Constants.PARTIAL_WEEK, HistoryMeasure]
        ]
        Actuals_identify[o9Constants.FORECAST_ITERATION] = the_iteration
        AssignedAlgoList[o9Constants.FORECAST_ITERATION] = the_iteration
        ValidationParameters[o9Constants.FORECAST_ITERATION] = the_iteration
        Weights[o9Constants.FORECAST_ITERATION] = the_iteration
        ForecastEngine[o9Constants.FORECAST_ITERATION] = the_iteration
        Violations[o9Constants.FORECAST_ITERATION] = the_iteration
        PlanningCycleDates[o9Constants.FORECAST_ITERATION] = the_iteration

        (
            bestfit_algo,
            validation_error,
            the_validation_error_planningcycle_pl,
            the_validation_forecast_pl,
        ) = identify_best_fit_mode(
            Grains=Grains,
            HistoryMeasure=HistoryMeasure,
            TimeDimension=TimeDimension,
            ForecastParameters=ForecastParameters,
            CurrentTimePeriod=CurrentTimePeriod,
            Actuals=Actuals_identify,
            ForecastData=all_ensemble_fcst_identify,
            OverrideFlatLineForecasts=OverrideFlatLineForecasts,
            ForecastGenTimeBucket=ForecastGenTimeBucket,
            AssignedAlgoList=AssignedAlgoList,
            SelectionCriteria=SelectionCriteria,
            multiprocessing_num_cores=multiprocessing_num_cores,
            df_keys=df_keys,
            Weights=Weights,
            Violations=Violations,
            MasterAlgoList=MasterAlgoList,
            ForecastEngine=ForecastEngine,
            SellOutOffset=SellOutOffset,
            PlanningCycleDates=PlanningCycleDates,
            ValidationParameters=ValidationParameters,
        )

        # system bestfit algo is Ensemble if fcst strategy is Ensemble
        if forecast_strategy == "Ensemble":
            bestfit_algo[o9Constants.SYSTEM_BESTFIT_ALGORITHM] = "Ensemble"

        # output only Ensemble algo
        bestfit_algo = bestfit_algo[
            bestfit_algo[o9Constants.SYSTEM_BESTFIT_ALGORITHM] == "Ensemble"
        ]
        validation_error = validation_error[
            validation_error[o9Constants.STAT_ALGORITHM] == "Ensemble"
        ]

        # Join with algo list
        relevant_cols = forecast_level + [
            o9Constants.ASSIGNED_RULE,
            o9Constants.PLANNER_BESTFIT_ALGORITHM,
            o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST,
            o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST,
        ]
        validation_error = validation_error.merge(
            AssignedAlgoList[relevant_cols], on=forecast_level, how="inner"
        )

        validation_error = validation_error.merge(system_ensemble_algo_list, how="inner")

        validation_error[o9Constants.STAT_RULE] = validation_error.apply(get_rule, axis=1)

        # getting stat rule
        all_ensemble_desc_df = all_ensemble_desc_df.merge(
            validation_error[forecast_level + [o9Constants.STAT_RULE]]
        )

        ForecastData = ForecastData.merge(all_ensemble_fcst_df)

        # check if any values are not in master list
        missing_algos_Segmentation = [
            x
            for x in SegmentationOutput[o9Constants.BESTFIT_ALGORITHM].unique()
            if x not in list(set(custom_algo_list).union(set(get_algo_ranking())))
        ]

        if missing_algos_Segmentation:
            logger.warning(
                f"Please add {missing_algos_Segmentation} to the get_algo_ranking() function and run again"
            )

            # filter out to avoid downstream issues
            SegmentationOutput = SegmentationOutput[
                ~SegmentationOutput[o9Constants.BESTFIT_ALGORITHM].isin(missing_algos_Segmentation)
            ]

        # check if any values are not in master list
        missing_algos_AlgoSTats = [
            x
            for x in AlgoStats[o9Constants.STAT_ALGORITHM].unique()
            if x not in list(set(custom_algo_list).union(set(get_algo_ranking())))
        ]

        if missing_algos_AlgoSTats:
            logger.warning(
                f"Please add {missing_algos_AlgoSTats} to the get_algo_ranking() function and run again"
            )

            # filter out to avoid downstream issues
            AlgoStats = AlgoStats[
                ~AlgoStats[o9Constants.STAT_ALGORITHM].isin(missing_algos_AlgoSTats)
            ]

        SegmentationOutput = SegmentationOutput.merge(system_ensemble_algo_list, how="inner")
        if SegmentationOutput.empty:
            logger.warning(
                "Empty Dataframe for SegmentationOuput after merging with relevant ensemble intersection ..."
            )
            logger.warning(f"Returning empty dataframe for slice {df_keys} ...")
            return (
                BestFitAlgo,
                ValidationError,
                FcstNextNBuckets,
                AllEnsembleDesc,
                AllEnsembleFcst,
                SystemEnsembleAlgorithmList,
                OutputAllAlgo,
            )

        # updating bestfit algorithm
        SegmentationOutput = SegmentationOutput.merge(
            bestfit_algo[forecast_level + [o9Constants.SYSTEM_BESTFIT_ALGORITHM]],
            how="left",
        )
        SegmentationOutput[o9Constants.BESTFIT_ALGORITHM] = np.where(
            ~SegmentationOutput[o9Constants.SYSTEM_BESTFIT_ALGORITHM].isna(),
            SegmentationOutput[o9Constants.SYSTEM_BESTFIT_ALGORITHM],
            SegmentationOutput[o9Constants.BESTFIT_ALGORITHM],
        )
        SegmentationOutput.drop(o9Constants.SYSTEM_BESTFIT_ALGORITHM, axis=1, inplace=True)

        # getting ensemble algo stats
        validation_method = AlgoStats[o9Constants.VALIDATION_METHOD].iloc[
            0
        ]  # getting validation method
        AlgoStats = all_ensemble_desc_df.merge(
            validation_error[
                [o9Constants.VERSION_NAME]
                + forecast_level
                + [
                    o9Constants.STAT_ALGORITHM,
                    o9Constants.STAT_RULE,
                    o9Constants.VALIDATION_ERROR,
                    o9Constants.COMPOSITE_ERROR,
                ]
            ]
        )
        AlgoStats[o9Constants.VALIDATION_METHOD] = validation_method

        # copying ensemble results in stst fcst l1 measure
        ForecastData[stat_fcst_l1_col] = ForecastData[stat_fcst_ensemble_col]

        (
            OutputAllAlgo,
            OutputBestFit,
            ActualLastNBuckets,
            FcstNextNBuckets,
            AlgoStatsForBestFitMembers,
        ) = output_checks(
            Actual=Actuals,
            ForecastData=ForecastData,
            SegmentationOutput=SegmentationOutput,
            TimeDimension=TimeDimension,
            CurrentTimePeriod=CurrentTimePeriod,
            ForecastGenTimeBucket=ForecastGenTimeBucket,
            ForecastSetupConfiguration=ForecastSetupConfiguration,
            AlgoStats=AlgoStats,
            TrendVariationThreshold=TrendVariationThreshold,
            LevelVariationThreshold=LevelVariationThreshold,
            RangeVariationThreshold=RangeVariationThreshold,
            SeasonalVariationThreshold=SeasonalVariationThreshold,
            SeasonalVariationCountThreshold=SeasonalVariationCountThreshold,
            ReasonabilityCycles=ReasonabilityCycles,
            MinimumIndicePercentage=MinimumIndicePercentage,
            AbsTolerance=AbsTolerance,
            Grains=Grains,
            df_keys=df_keys,
            COCCVariationThreshold=COCCVariationThreshold,
            SellOutOffset=SellOutOffset,
        )

        OutputAllAlgo[o9Constants.VALIDATION_METHOD] = validation_method
        OutputAllAlgo = OutputAllAlgo[OutputAllAlgo[o9Constants.STAT_ALGORITHM] == "Ensemble"]
        FcstNextNBuckets = FcstNextNBuckets[
            FcstNextNBuckets[o9Constants.STAT_ALGORITHM] == "Ensemble"
        ]

        BestFitAlgo = bestfit_algo[BestFitAlgo_cols]
        ValidationError = validation_error[ValidationError_cols]
        FcstNextNBuckets = FcstNextNBuckets[FcstNextNBuckets_cols]
        AllEnsembleDesc = all_ensemble_desc_df[cols_required_in_all_ensemble_desc_df]
        AllEnsembleFcst = all_ensemble_fcst_df[cols_required_in_all_ensemble_fcst_df]
        SystemEnsembleAlgorithmList = system_ensemble_algo_list[
            cols_required_in_system_ensemble_algo_list
        ]
        OutputAllAlgo = OutputAllAlgo[OutputAllAlgo_cols]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
    return (
        BestFitAlgo,
        ValidationError,
        FcstNextNBuckets,
        AllEnsembleDesc,
        AllEnsembleFcst,
        SystemEnsembleAlgorithmList,
        OutputAllAlgo,
    )
