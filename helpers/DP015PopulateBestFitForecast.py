from functools import reduce

import numpy as np
import pandas as pd
import polars as pl
from o9Reference.common_utils.common_utils import (
    get_last_time_period_polars,
    get_n_time_periods_polars,
)
from o9Reference.common_utils.dataframe_utils import (
    add_columns_to_df_polars,
    concat_to_dataframe_polars,
)
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data_polars

from helpers.o9Constants import o9Constants
from helpers.o9helpers.o9logger import O9Logger
from helpers.utils import filter_for_iteration_polars

logger = O9Logger()
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

col_mapping = {
    "System Stat Fcst L1": float,
    "System Stat Fcst L1 80% UB": float,
    "System Stat Fcst L1 80% LB": float,
    "Straight Line": float,
    "Trend Violation": float,
    "Level Violation": float,
    "Seasonal Violation": float,
    "Range Violation": float,
    "Run Count": float,
    "No Alerts": float,
    "Is Bestfit": float,
    "Algorithm Parameters": str,
    "Fcst Next N Buckets": float,
    "Run Time": float,
    "Validation Error": float,
    "Validation Method": str,
    "Validation Actual": float,
    "Validation Fcst Abs Error": float,
    "Validation Fcst": float,
    "Bestfit Straight Line": float,
    "Bestfit Trend Violation": float,
    "Bestfit Level Violation": float,
    "Bestfit Seasonal Violation": float,
    "Bestfit Range Violation": float,
    "Missing Bestfit": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    TimeDimension,
    ForecastParameters,
    CurrentTimePeriod,
    ForecastData,
    ForecastBounds,
    BestFitAlgo,
    ForecastGenTimeBucket,
    StatBucketWeight,
    Violations=None,
    df_keys={},
):
    try:

        def convert_to_polars(df, name):
            try:
                logger.info("Converting to polars")
                return pl.from_pandas(df)
            except Exception as e:
                logger.error(f"Error converting {name} to Polars: {e}")
                return None

        TimeDimension = convert_to_polars(TimeDimension, "TimeDimension")
        ForecastParameters = convert_to_polars(ForecastParameters, "ForecastParameters")
        CurrentTimePeriod = convert_to_polars(CurrentTimePeriod, "CurrentTimePeriod")
        ForecastData = convert_to_polars(ForecastData, "ForecastData")
        ForecastBounds = convert_to_polars(ForecastBounds, "ForecastBounds")
        BestFitAlgo = convert_to_polars(BestFitAlgo, "BestFitAlgo")
        ForecastGenTimeBucket = convert_to_polars(ForecastGenTimeBucket, "ForecastGenTimeBucket")
        StatBucketWeight = convert_to_polars(StatBucketWeight, "StatBucketWeight")
        Violations = convert_to_polars(Violations, "Violations")

        BestFitForecastList = list()
        BestFitAlgorithmCandidateOutputList = list()
        BestFitViolationOutputList = list()

        for the_iteration in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration_polars(iteration=the_iteration)(processIteration)
            (
                the_bestfitforecast,
                the_bestfitalgocandidateoutput,
                the_bestfitviolationoutput,
            ) = decorated_func(
                Grains=Grains,
                TimeDimension=TimeDimension,
                ForecastParameters=ForecastParameters,
                CurrentTimePeriod=CurrentTimePeriod,
                ForecastData=ForecastData,
                ForecastBounds=ForecastBounds,
                BestFitAlgo=BestFitAlgo,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                StatBucketWeight=StatBucketWeight,
                Violations=Violations,
                df_keys=df_keys,
            )

            BestFitForecastList.append(the_bestfitforecast)
            BestFitAlgorithmCandidateOutputList.append(the_bestfitalgocandidateoutput)
            BestFitViolationOutputList.append(the_bestfitviolationoutput)

        BestFitForecast = concat_to_dataframe_polars(BestFitForecastList)
        BestFitAlgorithmCandidateOutput = concat_to_dataframe_polars(
            BestFitAlgorithmCandidateOutputList
        )
        BestFitViolationOutput = concat_to_dataframe_polars(BestFitViolationOutputList)
    except Exception as e:
        logger.exception(e)
        (
            BestFitForecast,
            BestFitAlgorithmCandidateOutput,
            BestFitViolationOutput,
        ) = (None, None, None)
    BestFitForecast = BestFitForecast.to_pandas()
    BestFitAlgorithmCandidateOutput = BestFitAlgorithmCandidateOutput.to_pandas()
    BestFitViolationOutput = BestFitViolationOutput.to_pandas()
    return (
        BestFitForecast,
        BestFitAlgorithmCandidateOutput,
        BestFitViolationOutput,
    )


class Constants:
    forecast_period_col = o9Constants.FORECAST_PERIOD
    validation_period_col = o9Constants.VALIDATION_PERIOD
    history_time_buckets_col = o9Constants.HISTORY_TIME_BUCKETS
    error_metric_col = o9Constants.ERROR_METRIC
    history_period_col = o9Constants.HISTORY_PERIOD
    best_fit_method_col = o9Constants.BESTFIT_METHOD
    system_best_fit_algo_col = o9Constants.SYSTEM_BESTFIT_ALGORITHM
    planner_best_fit_algo_col = o9Constants.PLANNER_BESTFIT_ALGOITHM
    system_stat_fcst_l1_col = o9Constants.SYSTEM_STAT_FCST_L1
    best_fit_ub_algo_col = o9Constants.SYSTEM_BEST_FIT_ALGO_COL_UB
    best_fit_lb_algo_col = o9Constants.SYSTEM_BEST_FIT_ALGO_COL_LB
    system_stat_fcst_l1_ub_col = o9Constants.SYSTEM_STAT_FCST_L1_COL_UB
    system_stat_fcst_l1_lb_col = o9Constants.SYSTEM_STAT_FCST_L1_COL_LB
    stat_fcst_str = o9Constants.STAT_FCST + " "
    week_col = o9Constants.WEEK
    month_col = o9Constants.MONTH
    planning_month_col = o9Constants.PLANNING_MONTH
    quarter_col = o9Constants.QUARTER
    planning_quarter_col = o9Constants.PLANNING_QUARTER
    week_key_col = o9Constants.WEEK_KEY
    month_key_col = o9Constants.MONTH_KEY
    planning_month_key_col = o9Constants.PLANNING_MONTH_KEY
    quarter_key_col = o9Constants.QUARTER_KEY
    planning_quarter_key_col = o9Constants.PLANNING_QUARTER_KEY
    partial_week_col = o9Constants.PARTIAL_WEEK
    version_col = o9Constants.VERSION_NAME
    STAT_ALGO: str = o9Constants.STAT_ALGORITHM
    fcst_gen_time_bucket_col = o9Constants.FORECAST_GEN_TIME_BUCKET
    stat_bucket_weight_col = o9Constants.STAT_BUCKET_WEIGHT
    STRAIGHT_LINE: str = o9Constants.STRAIGHT_LINE
    TREND_VIOLATION: str = o9Constants.TREND_VIOLATION
    LEVEL_VIOLATION: str = o9Constants.LEVEL_VIOLATION
    SEASONAL_VIOLATION: str = o9Constants.SEASONAL_VIOLATION
    RANGE_VIOLATION: str = o9Constants.RANGE_VIOLATION
    COCC_VIOLATION: str = o9Constants.COCC_VIOLATION
    RUN_COUNT: str = o9Constants.RUN_COUNT
    NO_ALERTS: str = o9Constants.NO_ALERTS
    IS_BESTFIT: str = o9Constants.IS_BESTFIT
    ALGORITHM_PARAMETERS: str = o9Constants.ALGORITHM_PARAMETERS
    FCST_NEXT_N_BUCKETS: str = o9Constants.FCST_NEXT_N_BUCKETS
    RUN_TIME: str = o9Constants.RUN_TIME
    VALIDATION_ERROR: str = o9Constants.VALIDATION_ERROR
    VALIDATION_METHOD: str = o9Constants.VALIDATION_METHOD
    COMPOSITE_ERROR: str = o9Constants.COMPOSITE_ERROR
    VALIDATION_ACTUAL: str = o9Constants.VALIDATION_ACTUAL
    VALIDATION_FCST_ABS_ERROR: str = o9Constants.VALIDATION_FCST_ABS_ERROR
    VALIDATION_FCST: str = o9Constants.VALIDATION_FCST
    BESTFIT_STRAIGHT_LINE: str = o9Constants.BESTFIT_STRAIGHT_LINE
    BESTFIT_TREND_VIOLATION: str = o9Constants.BESTFIT_TREND_VIOLATION
    BESTFIT_LEVEL_VIOLATION: str = o9Constants.BESTFIT_LEVEL_VIOLATION
    BESTFIT_SEASONAL_VIOLATION: str = o9Constants.BESTFIT_SEASONAL_VIOLATION
    BESTFIT_RANGE_VIOLATION: str = o9Constants.BESTFIT_RANGE_VIOLATION
    BESTFIT_COCC_VIOLATION: str = o9Constants.BESTFIT_COCC_VIOLATION
    MISSING_BESTFIT: str = o9Constants.MISSING_BESTFIT

    week = "Week"
    planning_month = "Planning Month"
    month = "Month"
    planning_quarter = "Planning Quarter"
    quarter = "Quarter"

    forecast_parameter_req_cols = [
        version_col,
        history_period_col,
        forecast_period_col,
        validation_period_col,
        best_fit_method_col,
        error_metric_col,
        history_time_buckets_col,
    ]
    bestfit_violation_columns = [
        BESTFIT_STRAIGHT_LINE,
        BESTFIT_TREND_VIOLATION,
        BESTFIT_LEVEL_VIOLATION,
        BESTFIT_SEASONAL_VIOLATION,
        BESTFIT_RANGE_VIOLATION,
        BESTFIT_COCC_VIOLATION,
        MISSING_BESTFIT,
    ]
    bestfit_algorithm_candidate_columns = [
        o9Constants.STAT_RULE,
        STAT_ALGO,
        STRAIGHT_LINE,
        TREND_VIOLATION,
        LEVEL_VIOLATION,
        SEASONAL_VIOLATION,
        RANGE_VIOLATION,
        COCC_VIOLATION,
        RUN_COUNT,
        NO_ALERTS,
        IS_BESTFIT,
        ALGORITHM_PARAMETERS,
        FCST_NEXT_N_BUCKETS,
        RUN_TIME,
        VALIDATION_ERROR,
        VALIDATION_METHOD,
        COMPOSITE_ERROR,
        VALIDATION_ACTUAL,
        VALIDATION_FCST_ABS_ERROR,
        VALIDATION_FCST,
    ]
    required_output_columns = [
        partial_week_col,
        system_stat_fcst_l1_col,
        system_stat_fcst_l1_ub_col,
        system_stat_fcst_l1_lb_col,
    ]

    time_bucket = [week, planning_month, month, planning_quarter, quarter]


def get_forecast_details(fcst_gen_time_bucket):
    config = {
        Constants.week: {
            "frequency": Constants.week + "ly",
            "cols": [
                Constants.partial_week_col,
                Constants.week_col,
                Constants.week_key_col,
            ],
            "name": Constants.week_col,
            "key": Constants.week_key_col,
        },
        Constants.planning_month: {
            "frequency": Constants.month + "ly",
            "cols": [
                Constants.partial_week_col,
                Constants.planning_month_col,
                Constants.planning_month_key_col,
            ],
            "name": Constants.planning_month_col,
            "key": Constants.planning_month_key_col,
        },
        Constants.month: {
            "frequency": Constants.month + "ly",
            "cols": [
                Constants.partial_week_col,
                Constants.month_col,
                Constants.month_key_col,
            ],
            "name": Constants.month_col,
            "key": Constants.month_key_col,
        },
        Constants.planning_quarter: {
            "frequency": Constants.quarter + "ly",
            "cols": [
                Constants.partial_week_col,
                Constants.planning_quarter_col,
                Constants.planning_quarter_key_col,
            ],
            "name": Constants.planning_quarter_col,
            "key": Constants.planning_quarter_key_col,
        },
        Constants.quarter: {
            "frequency": Constants.quarter + "ly",
            "cols": [
                Constants.partial_week_col,
                Constants.quarter_col,
                Constants.quarter_key_col,
            ],
            "name": Constants.quarter_col,
            "key": Constants.quarter_key_col,
        },
    }
    fcst_gen_time_bucket_config = config.get(fcst_gen_time_bucket)
    if fcst_gen_time_bucket_config is None:
        raise ValueError(f"Unknown forecast time bucket: {fcst_gen_time_bucket}")
    return (
        fcst_gen_time_bucket_config["frequency"],
        fcst_gen_time_bucket_config["cols"],
        fcst_gen_time_bucket_config["name"],
        fcst_gen_time_bucket_config["key"],
    )


def processIteration(
    Grains,
    TimeDimension,
    ForecastParameters,
    CurrentTimePeriod,
    ForecastData,
    ForecastBounds,
    BestFitAlgo,
    ForecastGenTimeBucket,
    StatBucketWeight,
    Violations=None,
    df_keys={},
):
    plugin_name = "DP015PopulateBestFitForecast"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables

    logger.info("Extracting forecast level ...")

    forecast_level = [x.strip() for x in Grains.split(",") if x.strip() not in ("NA", "")]

    common_cols = [Constants.version_col] + forecast_level
    cols_required_in_output = common_cols + Constants.required_output_columns
    cols_required_in_bestfit_algorithm_candidate_output = (
        common_cols + Constants.bestfit_algorithm_candidate_columns
    )
    cols_required_in_bestfit_violation_output = common_cols + Constants.bestfit_violation_columns

    BestFitForecast = pl.DataFrame({col: [] for col in cols_required_in_output})
    BestFitAlgorithmCandidateOutput = pl.DataFrame(
        {col: [] for col in cols_required_in_bestfit_algorithm_candidate_output}
    )
    BestFitViolationOutput = pl.DataFrame(
        {col: [] for col in cols_required_in_bestfit_violation_output}
    )
    try:
        ForecastParameters = ForecastParameters.select(Constants.forecast_parameter_req_cols)

        stat_fcst_cols = [x for x in ForecastData.columns if "Stat Fcst" in x]
        stat_fcst_bound_cols = [x for x in ForecastBounds.columns if "80%" in x]
        req_cols = common_cols + [Constants.partial_week_col] + stat_fcst_cols

        ForecastData = ForecastData.select(req_cols)

        ForecastData = ForecastData.filter(
            pl.any_horizontal([pl.col(c).is_not_null() for c in stat_fcst_cols])
        )
        ForecastBounds = ForecastBounds.filter(
            pl.any_horizontal([pl.col(c).is_not_null() for c in stat_fcst_bound_cols])
        )

        if Constants.system_best_fit_algo_col in BestFitAlgo.columns:
            pass
        elif o9Constants.SYSTEM_BESTFIT_ALGORITHM_FINAL in BestFitAlgo.columns:
            logger.info(
                f"Using {o9Constants.SYSTEM_BESTFIT_ALGORITHM_FINAL} for populating best fit forecast ..."
            )
            Constants.system_best_fit_algo_col = o9Constants.SYSTEM_BESTFIT_ALGORITHM_FINAL
        else:
            raise ValueError(
                "Either of System Bestfit Algorithm or System Bestfit Algorithm Final should be provided as input .."
            )

        bestfit_algo_req_cols = common_cols + [
            Constants.system_best_fit_algo_col,
            Constants.planner_best_fit_algo_col,
        ]

        if len(BestFitAlgo) == 0:
            logger.warning("BestFitAlgo is empty for slice : {}".format(df_keys))
            logger.warning("Returning empty dataframe ...")
            return (
                BestFitForecast,
                BestFitAlgorithmCandidateOutput,
                BestFitViolationOutput,
            )

        if Violations is None:
            Violations = pl.DataFrame()
            logger.info(f"Violations empty for slice {df_keys}")

        BestFitAlgo = BestFitAlgo.select(bestfit_algo_req_cols)
        input_version = BestFitAlgo.select(Constants.version_col).unique()[0]

        BestFitAlgo = BestFitAlgo.filter(pl.col(Constants.system_best_fit_algo_col).is_not_null())
        # Use planner best fit algo wherever available
        BestFitAlgo = BestFitAlgo.with_columns(
            pl.col(Constants.planner_best_fit_algo_col).fill_null(
                pl.col(Constants.system_best_fit_algo_col)
            )
        )

        # Drop system best fit column since not needed further
        BestFitAlgo = BestFitAlgo.drop(
            [Constants.system_best_fit_algo_col, Constants.version_col]
        ).drop_nulls()

        if TimeDimension.height == 0:
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return (
                BestFitForecast,
                BestFitAlgorithmCandidateOutput,
                BestFitViolationOutput,
            )

        # infer time related attributes from forecast gen time bucket
        fcst_gen_time_bucket = ForecastGenTimeBucket[Constants.fcst_gen_time_bucket_col][0]
        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        if fcst_gen_time_bucket in Constants.time_bucket:
            frequency, relevant_time_cols, relevant_time_name, relevant_time_key = (
                get_forecast_details(fcst_gen_time_bucket=fcst_gen_time_bucket)
            )
        else:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return (
                BestFitForecast,
                BestFitAlgorithmCandidateOutput,
                BestFitViolationOutput,
            )

        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")

        # retain time mapping with partial week
        base_time_mapping = TimeDimension.select(relevant_time_cols).unique()

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension.select(
            [relevant_time_name, relevant_time_key]
        ).unique()

        # Join Actuals with time mapping
        ForecastData = ForecastData.join(
            base_time_mapping, on=Constants.partial_week_col, how="inner"
        )

        # select the relevant columns, groupby and sum history measure
        ForecastData = ForecastData.group_by(forecast_level + [relevant_time_name]).agg(
            [pl.col(col).sum() for col in stat_fcst_cols]
        )
        forecast_periods = int(
            ForecastParameters.select(Constants.forecast_period_col).to_series()[0]
        )

        logger.info("CurrentTimePeriod head : ")
        logger.info(CurrentTimePeriod.head())

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # get the validation periods
        # note the negative sign to history periods
        latest_time_name = get_last_time_period_polars(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )

        # get the forecast period dates
        forecast_period_dates = get_n_time_periods_polars(
            latest_time_name,
            forecast_periods,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=False,
        )

        logger.info("forecast_period_dates : {}".format(forecast_period_dates))

        # filter forecast data
        AllForecast = ForecastData.filter(pl.col(relevant_time_name).is_in(forecast_period_dates))

        if AllForecast.height == 0:
            logger.warning(
                "No data found after filtering for forecast_period_dates for slice : {}".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframe ...")
            return (
                BestFitForecast,
                BestFitAlgorithmCandidateOutput,
                BestFitViolationOutput,
            )

        # merge best fit columns to master df containing all forecast
        AllForecast = AllForecast.join(BestFitAlgo, how="left", on=forecast_level)

        # if best fit is not available for any of the combinations, exit out
        if (
            AllForecast.select(pl.col(Constants.planner_best_fit_algo_col).is_null())
            .to_series()
            .all()
        ):
            logger.warning(
                "Best fit algorithm not present for any combination for slice : {}".format(df_keys)
            )
            logger.warning("Returning empty dataframe")
            return (
                BestFitForecast,
                BestFitAlgorithmCandidateOutput,
                BestFitViolationOutput,
            )

        # Check best fit algorithms is NA in any of the intersections
        intersections_with_no_bestfit = (
            AllForecast.filter(pl.col(Constants.planner_best_fit_algo_col).is_null())
            .select(forecast_level)
            .unique()
        )

        if len(intersections_with_no_bestfit) > 0:
            logger.warning(
                "Best fit algorithm is missing for {} intersections, printing top 5, for slice : {} ...".format(
                    len(intersections_with_no_bestfit), df_keys
                )
            )
            logger.warning(intersections_with_no_bestfit.head())
            logger.warning(
                "intersections_with_no_bestfit, shape : {}".format(
                    intersections_with_no_bestfit.shape
                )
            )

        # Filter rows where best fit columns is not NA
        AllForecast = AllForecast.filter(pl.col(Constants.planner_best_fit_algo_col).is_not_null())

        # Update the column by prefixing with stat_fcst_str
        AllForecast = AllForecast.with_columns(
            (
                pl.lit(Constants.stat_fcst_str)
                + pl.col(Constants.planner_best_fit_algo_col).cast(pl.Utf8)
            ).alias(Constants.planner_best_fit_algo_col)
        )

        # Get all unique column names referenced in planner_best_fit_algo_col
        choices = (
            AllForecast.select(pl.col(Constants.planner_best_fit_algo_col))
            .unique()
            .to_series()
            .to_list()
        )
        # Build the expression
        expr = reduce(
            lambda acc, colname: acc.when(
                pl.col(Constants.planner_best_fit_algo_col) == colname
            ).then(pl.col(colname)),
            choices[1:],
            pl.when(pl.col(Constants.planner_best_fit_algo_col) == choices[0]).then(
                pl.col(choices[0])
            ),
        ).otherwise(None)

        # Add the new column
        AllForecast = AllForecast.with_columns(expr.alias(Constants.system_stat_fcst_l1_col))

        AllForecast = AllForecast.with_columns(
            [
                (
                    pl.col(Constants.planner_best_fit_algo_col).cast(pl.Utf8) + pl.lit(" 80% UB")
                ).alias(Constants.best_fit_ub_algo_col),
                (
                    pl.col(Constants.planner_best_fit_algo_col).cast(pl.Utf8) + pl.lit(" 80% LB")
                ).alias(Constants.best_fit_lb_algo_col),
            ]
        )

        if ForecastBounds.height == 0:
            logger.warning(
                "ForecastBounds dataframe is empty for slice {}, will populate NaN into {}/{} columns".format(
                    df_keys,
                    Constants.system_stat_fcst_l1_ub_col,
                    Constants.system_stat_fcst_l1_lb_col,
                )
            )
            # if bounds are empty, populate NaNs into the 80 % LB and UB Columns
            AllForecast = AllForecast.with_columns(
                [
                    pl.lit(np.nan).alias(Constants.system_stat_fcst_l1_ub_col),
                    pl.lit(np.nan).alias(Constants.system_stat_fcst_l1_lb_col),
                ]
            )
        else:
            # Join Actuals with time mapping
            ForecastBounds = ForecastBounds.join(
                base_time_mapping, on=Constants.partial_week_col, how="inner"
            )
            groupby_cols = forecast_level + [relevant_time_name]

            # select the relevant columns, groupby and sum history measure
            ForecastBounds = ForecastBounds.group_by(groupby_cols).agg(
                [pl.sum(col) for col in stat_fcst_bound_cols]
            )

            # Join forecast bounds with forecasts dataframe
            AllForecast = AllForecast.join(
                ForecastBounds,
                on=groupby_cols,
                how="left",
            )

            # When running through backtest, it can happen that some of the bound columns are missing in the dataframe
            # This is because the algorithms cannot generate bounds and we are dropping the entire column if all rows are blank in the column
            # Populate missing columns for UB and LB with np.nan so that lookup does not fail
            AllForecast = add_columns_to_df_polars(
                AllForecast, list(AllForecast[Constants.best_fit_ub_algo_col].unique())
            )
            AllForecast = add_columns_to_df_polars(
                AllForecast, list(AllForecast[Constants.best_fit_lb_algo_col].unique())
            )

            # Get all unique column names referenced in best_fit_ub_algo_col and best_fit_lb_algo_col
            ub_choices = (
                AllForecast.select(pl.col(Constants.best_fit_ub_algo_col))
                .unique()
                .to_series()
                .to_list()
            )
            lb_choices = (
                AllForecast.select(pl.col(Constants.best_fit_lb_algo_col))
                .unique()
                .to_series()
                .to_list()
            )

            # Build lookup expr for UB
            ub_expr = reduce(
                lambda acc, colname: acc.when(
                    pl.col(Constants.best_fit_ub_algo_col) == colname
                ).then(pl.col(colname)),
                ub_choices[1:],
                pl.when(pl.col(Constants.best_fit_ub_algo_col) == ub_choices[0]).then(
                    pl.col(ub_choices[0])
                ),
            ).otherwise(None)

            # Build lookup expr for LB
            lb_expr = reduce(
                lambda acc, colname: acc.when(
                    pl.col(Constants.best_fit_lb_algo_col) == colname
                ).then(pl.col(colname)),
                lb_choices[1:],
                pl.when(pl.col(Constants.best_fit_lb_algo_col) == lb_choices[0]).then(
                    pl.col(lb_choices[0])
                ),
            ).otherwise(None)

            # Assign new columns
            AllForecast = AllForecast.with_columns(
                [
                    ub_expr.alias(Constants.system_stat_fcst_l1_ub_col),
                    lb_expr.alias(Constants.system_stat_fcst_l1_lb_col),
                ]
            )
        cols_to_disaggregate = [
            Constants.system_stat_fcst_l1_col,
            Constants.system_stat_fcst_l1_lb_col,
            Constants.system_stat_fcst_l1_ub_col,
        ]
        cols_required_at_relevant_time_level = (
            forecast_level + [relevant_time_name] + cols_to_disaggregate
        )
        # Collect only the required columns
        BestFitForecast = AllForecast.select(cols_required_at_relevant_time_level)

        # get statbucket weights at the desired level
        StatBucketWeight = StatBucketWeight.join(
            base_time_mapping, on=Constants.partial_week_col, how="inner"
        )

        # perform disaggregation
        BestFitForecast = disaggregate_data_polars(
            source_df=BestFitForecast,
            source_grain=relevant_time_name,
            target_grain=Constants.partial_week_col,
            profile_df=StatBucketWeight,
            profile_col=Constants.stat_bucket_weight_col,
            cols_to_disaggregate=cols_to_disaggregate,
        )
        BestFitForecast = BestFitForecast.with_columns(
            [pl.lit(input_version[Constants.version_col][0]).alias(Constants.version_col)]
        )

        # re order columns
        BestFitForecast = BestFitForecast.select(cols_required_in_output)

        if Violations.height > 0:
            logger.info("Calculating violations for best fit algorithm ...")

            # populate the violation attributes against the best fit algorithm
            Violations = Violations.join(BestFitAlgo, on=forecast_level, how="left")
            # identify cases where bestfit missing
            Violations = Violations.with_columns(
                pl.when(pl.col(Constants.planner_best_fit_algo_col).is_null())
                .then(1)
                .otherwise(0)
                .alias(Constants.MISSING_BESTFIT)
            )

            # filter out the records where algorithm column is same as best fit algo
            BestFitAlgorithmCandidateOutput_raw = Violations.filter(
                pl.col(Constants.STAT_ALGO) == pl.col(Constants.planner_best_fit_algo_col)
            )

            # Step 2: Add the IS_BESTFIT column with value 1.0
            BestFitAlgorithmCandidateOutput_raw = BestFitAlgorithmCandidateOutput_raw.with_columns(
                pl.lit(1.0).alias(Constants.IS_BESTFIT)
            )

            BestFitAlgorithmCandidateOutput_bestfitalgorithm = (
                BestFitAlgorithmCandidateOutput_raw.with_columns(
                    pl.lit("Bestfit Algorithm").alias(Constants.STAT_ALGO)
                )
            )

            BestFitAlgorithmCandidateOutput = pl.concat(
                [
                    BestFitAlgorithmCandidateOutput_raw,
                    BestFitAlgorithmCandidateOutput_bestfitalgorithm,
                ]
            )

            BestFitAlgorithmCandidateOutput = BestFitAlgorithmCandidateOutput.select(
                cols_required_in_bestfit_algorithm_candidate_output
            )

            rename_mapping = {
                Constants.STRAIGHT_LINE: Constants.BESTFIT_STRAIGHT_LINE,
                Constants.TREND_VIOLATION: Constants.BESTFIT_TREND_VIOLATION,
                Constants.LEVEL_VIOLATION: Constants.BESTFIT_LEVEL_VIOLATION,
                Constants.SEASONAL_VIOLATION: Constants.BESTFIT_SEASONAL_VIOLATION,
                Constants.RANGE_VIOLATION: Constants.BESTFIT_RANGE_VIOLATION,
                Constants.COCC_VIOLATION: Constants.BESTFIT_COCC_VIOLATION,
            }

            BestFitViolationOutput = Violations.filter(
                (pl.col(Constants.STAT_ALGO) == pl.col(Constants.planner_best_fit_algo_col))
                | (pl.col(Constants.planner_best_fit_algo_col).is_null())
            )

            BestFitViolationOutput = BestFitViolationOutput.rename(rename_mapping)
            BestFitViolationOutput = BestFitViolationOutput[
                cols_required_in_bestfit_violation_output
            ].unique()

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        BestFitForecast = pl.DataFrame({col: [] for col in cols_required_in_output})
        BestFitAlgorithmCandidateOutput = pl.DataFrame(
            {col: [] for col in cols_required_in_bestfit_algorithm_candidate_output}
        )
        BestFitViolationOutput = pl.DataFrame(
            {col: [] for col in cols_required_in_bestfit_violation_output}
        )
    return (
        BestFitForecast,
        BestFitAlgorithmCandidateOutput,
        BestFitViolationOutput,
    )
