import logging
from typing import Dict, List, Tuple, Union

import pandas as pd
import polars as pl
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from utilsforecast.losses import mape, rmse

from helpers.o9Constants import o9Constants
from helpers.utils_polars import (
    build_iter_inputs_from_dict,
    get_last_time_period,
    get_n_time_periods,
)

logger = logging.getLogger("o9_logger")


def get_list_of_grains_from_string(input_str: str) -> List[str]:
    """Gets a list of grain columns from a comma-separated string."""
    return [s.strip() for s in input_str.split(",")] if input_str else []


def calculate_planning_cycle_dates(
    validation_params_df: pl.DataFrame,
    PlanningCycleDates_pl: pl.DataFrame,
    current_time_period_df: pl.DataFrame,
    time_dimension_df: pl.DataFrame,
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
        if col_name in validation_params_df.columns:
            return validation_params_df.select(col_name).item()
        return None

    period = get_param("Validation Period")
    fold = get_param("Validation Fold")
    step = get_param("Validation Step Size")

    if not all([period, fold, step]):
        logger.warning("Missing validation parameters. Cannot generate cycles.")
        return []

    cycles_list = list(range(int(period), int(period) + (int(fold) * int(step)), +int(step)))
    if not cycles_list:
        logger.warning("No validation cycles generated; cannot calculate dates.")
        return []

    cycles_df = pl.DataFrame({"cycle": sorted(list(set(cycles_list)))})

    # Get the latest time value
    latest_time_value = current_time_period_df.select(time_name_col).item()

    def get_date_for_cycle(cycle: int) -> Union[str, None]:
        """Safely gets the date for a single cycle number."""
        date_list = get_n_time_periods(
            latest_value=latest_time_value,
            periods=-int(cycle + offset_period),
            time_mapping=time_dimension_df,
            time_name_col=time_name_col,
            time_key_col=time_key_col,
            include_latest_value=False,
        )
        return date_list[0] if date_list else None

    planning_cycles = (
        cycles_df.select(
            pl.col("cycle")
            .map_elements(get_date_for_cycle, return_dtype=pl.String)
            .alias("planning_date")
        )
        .get_column("planning_date")
        .drop_nulls()
        .to_list()
    )

    time_dimension_df = time_dimension_df.with_columns(
        pl.col(o9Constants.PARTIAL_WEEK_KEY).alias("merge_key"),
    )

    planning_cycle_dates = (
        PlanningCycleDates_pl.join(
            time_dimension_df,
            left_on="Planning Cycle.[PlanningCycleDateKey]",
            right_on="merge_key",
            how="inner",
        )
        .unique(subset=["Planning Cycle.[PlanningCycleDateKey]"], keep="first")
        .sort(o9Constants.PARTIAL_WEEK_KEY)
        .select(
            pl.col(o9Constants.PLANNING_CYCLE_DATE), pl.col(time_name_col), pl.col(time_key_col)
        )
        .filter(pl.col(time_name_col).is_in(planning_cycles))
        .group_by(time_name_col, time_key_col)
        .agg(pl.col(o9Constants.PLANNING_CYCLE_DATE).first())
        .sort(time_key_col, descending=True)
        .get_column(o9Constants.PLANNING_CYCLE_DATE)
        .to_list()
    )

    return planning_cycle_dates


def get_algo_ranking() -> List[str]:
    """Gets the master list of algorithms in a preferred rank order to break ties."""
    ordered_list = [
        "Ensemble",
        "Moving Average",
        "SES",
        "DES",
        "Seasonal Naive YoY",
        "Simple Snaive",
        "Simple AOA",
        "Weighted Snaive",
        "Weighted AOA",
        "Naive Random Walk",
        "Croston",
        # "Croston TSB",
        "TES",
        "ETS",
        "STLF",
        "Theta",
        "Growth Snaive",
        "Growth AOA",
        "TBATS",
        "Auto ARIMA",
        "Prophet",
        "AR-NNET",
        "sARIMA",
        "SCHM",
        "CML",
    ]
    return ordered_list


def _prepare_base_data(
    df_actuals_forecasts: pl.DataFrame,
    df_time_dimension: pl.DataFrame,
    forecast_level: List[str],
    history_measure: str,
    time_config: Dict[str, str],
    all_stat_forecast_cols: List[str],
) -> pl.DataFrame:
    """Joins data and aggregates to the correct frequency."""
    list_of_measures_to_agg = [history_measure] + all_stat_forecast_cols
    return (
        df_actuals_forecasts.join(df_time_dimension, on=time_config["join_col"], how="inner")
        .group_by(forecast_level + [o9Constants.PLANNING_CYCLE_DATE, time_config["name"]])
        .agg(
            [
                (
                    pl.when(pl.col(col).is_not_null().any())
                    .then(pl.col(col).sum())
                    .otherwise(None)
                    .alias(col)
                )
                for col in list_of_measures_to_agg
            ]
        )
    )


def _calculate_validation_error(
    df: pl.DataFrame,
    history_measure: str,
    validation_error_col: str,
    stat_fcst_str: str,
    error_metric: str,
    override_flat_line: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    models = [c for c in df.columns if c.startswith(stat_fcst_str)]
    if not models:
        return pl.DataFrame(), pl.DataFrame()

    if override_flat_line:
        flat_line_exprs = [
            pl.when(pl.col(m).drop_nulls().n_unique() <= 1).then(None).otherwise(pl.col(m)).alias(m)
            for m in models
        ]
        aggregated = df.group_by(
            "unique_id", o9Constants.PLANNING_CYCLE_DATE, maintain_order=True
        ).agg(*flat_line_exprs, pl.col(history_measure))
        list_cols = [col for col, dtype in aggregated.schema.items() if "List" in str(dtype)]
        df = aggregated.explode(list_cols) if list_cols else aggregated

    cv_df = df.select(
        "unique_id",
        o9Constants.PLANNING_CYCLE_DATE,
        pl.col(history_measure).alias("y"),
        pl.all().exclude(["unique_id", history_measure, o9Constants.PLANNING_CYCLE_DATE]),
    )
    if cv_df.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    metric_func = rmse if error_metric == "RMSE" else mape

    results_list = [
        metric_func(group_df.drop_nulls("y"), models=models).with_columns(
            pl.lit(key[0]).alias(o9Constants.PLANNING_CYCLE_DATE)
        )
        for key, group_df in cv_df.group_by(o9Constants.PLANNING_CYCLE_DATE)
        if not group_df.drop_nulls("y").is_empty()
    ]

    if results_list:
        error_evals = pl.concat(results_list).select(
            [o9Constants.PLANNING_CYCLE_DATE, pl.all().exclude(o9Constants.PLANNING_CYCLE_DATE)]
        )
    else:
        error_evals = pl.DataFrame()

    if error_evals.is_empty():
        error_evals_planningcycle = pl.DataFrame()
    else:
        error_evals_planningcycle = (
            error_evals.unpivot(
                index=["unique_id", o9Constants.PLANNING_CYCLE_DATE],
                variable_name=o9Constants.STAT_ALGORITHM,
                value_name=validation_error_col,
            )
            .with_columns(
                pl.col(o9Constants.STAT_ALGORITHM)
                .str.replace("Stat Fcst ", "")
                .str.replace(" Planning Cycle", "")
            )
            .sort(["unique_id", o9Constants.PLANNING_CYCLE_DATE, o9Constants.STAT_ALGORITHM])
        )

    agg_cals = (
        df.unpivot(
            index=["unique_id", history_measure],
            on=models,
            variable_name=o9Constants.STAT_ALGORITHM,
            value_name="forecast",
        )
        .group_by(["unique_id", o9Constants.STAT_ALGORITHM])
        .agg(
            pl.col(history_measure).sum().alias("Validation Actual"),
            pl.col("forecast").sum().alias("Validation Fcst"),
            (pl.col(history_measure) - pl.col("forecast"))
            .abs()
            .sum()
            .alias("Validation Fcst Abs Error"),
        )
        .with_columns(
            pl.col(o9Constants.STAT_ALGORITHM)
            .str.replace("Stat Fcst ", "")
            .str.replace(" Planning Cycle", "")
        )
        .sort(["unique_id", o9Constants.STAT_ALGORITHM])
    )

    return agg_cals, error_evals_planningcycle


def _calculate_composite_error(
    df_error: pl.DataFrame,
    df_violations: pl.DataFrame,
    df_weights: pl.DataFrame,
    id_mapping_df: pl.DataFrame,
    forecast_level: List[str],
    config: Dict[str, Union[str, List[str]]],
) -> pl.DataFrame:
    """
    Calculates the composite error by merging validation errors with weighted violations.
    Includes error handling to ensure the pipeline continues if calculation fails.
    """
    try:
        df_violations = df_violations.join(id_mapping_df, on=forecast_level, how="inner").drop(
            forecast_level + [o9Constants.VERSION_NAME]
        )

        weighted_violations = (
            df_violations.join(df_weights, on=config["weights_join_cols"], how="left")
            .with_columns(
                *[
                    pl.col(v) * pl.col(w)
                    for v, w in zip(config["violation_cols"], config["weight_cols"])
                ]
            )
            .with_columns(pl.sum_horizontal(config["violation_cols"]).alias("Weighted Sum"))
        )

        composite_error_expr = (
            pl.when(pl.lit(config["error_metric"]) == "MAPE")
            .then(
                pl.when(pl.col(config["val_actual_col"]) == 0)
                .then(pl.col(config["val_error_col"]))
                .otherwise(
                    100
                    * (pl.col(config["val_abs_error_col"]) * (1 + pl.col("Weighted Sum")))
                    / pl.col(config["val_actual_col"])
                )
            )
            .when(pl.lit(config["error_metric"]) == "RMSE")
            .then(pl.col(config["val_error_col"]) * (1 + pl.col("Weighted Sum")))
            .otherwise(pl.col(config["val_error_col"]))  # Default case
        )

        return (
            df_error.join(
                weighted_violations.select(config["weighted_cols"] + ["Weighted Sum"]),
                on=config["composite_join_cols"],
                how="left",
                suffix="_PlannerOverride",
            )
            .with_columns(
                [
                    pl.col("Weighted Sum").fill_null(0),
                    # Replace Stat Rule with Planner Override if exists, else keep original
                    pl.when(pl.col(f"{o9Constants.STAT_RULE}_PlannerOverride").is_not_null())
                    .then(pl.col(f"{o9Constants.STAT_RULE}_PlannerOverride"))
                    .otherwise(pl.col(o9Constants.STAT_RULE))
                    .alias(o9Constants.STAT_RULE),
                    composite_error_expr.alias(config["composite_error_col"]),
                ]
            )
            .drop(["Weighted Sum", f"{o9Constants.STAT_RULE}_PlannerOverride"])
        )
    except Exception as e:
        logger.exception(
            f"Could not calculate composite error due to an error: {e}. "
            "Proceeding with validation error as the composite error."
        )
        return df_error.with_columns(
            pl.col(config["val_error_col"]).alias(config["composite_error_col"])
        )


def _select_best_fit(
    df_final_error: pl.DataFrame,
    forecast_level: List[str],
    composite_error_col: str,
    master_algo_list: List[str],
) -> pl.DataFrame:
    """Sorts by error and algorithm rank to select the best model for each grain."""
    algo_rank_df = pl.DataFrame(
        {o9Constants.STAT_ALGORITHM: master_algo_list, "algo_rank": range(len(master_algo_list))}
    )
    return (
        df_final_error.filter(pl.col(composite_error_col).is_not_null())
        .join(algo_rank_df, on=o9Constants.STAT_ALGORITHM, how="left")
        .sort(by=[forecast_level, composite_error_col, "algo_rank"])
        .group_by(forecast_level, maintain_order=True)
        .first()
        .rename({o9Constants.STAT_ALGORITHM: o9Constants.SYSTEM_BESTFIT_ALGORITHM})
    )


def run_best_fit_analysis_pl(
    Grains_str: str,
    HistoryMeasure_str: str,
    TimeDimension_pl: pl.DataFrame,
    ForecastParameters_pl: pl.DataFrame,
    CurrentTimePeriod_pl: pl.DataFrame,
    Actuals_pl: pl.DataFrame,
    ForecastData_pl: pl.DataFrame,
    ValidationParameters_pl: pl.DataFrame,
    OverrideFlatLineForecasts_str: str,
    ForecastGenTimeBucket_pl: pl.DataFrame,
    AssignedAlgoList_pl: pl.DataFrame,
    PlanningCycleDates_pl: pl.DataFrame,
    SelectionCriteria_pl: pl.DataFrame,
    MasterAlgoList_pl: pl.DataFrame,
    Weights_pl: pl.DataFrame,
    Violations_pl: pl.DataFrame,
    SellOutOffset_pl: pl.DataFrame,
    ForecastEngine_pl: pl.DataFrame,
    **kwargs,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Main Polars pipeline to identify the best fit forecasting model using integer IDs for optimization."""

    try:

        forecast_level = get_list_of_grains_from_string(Grains_str)
        stat_fcst_str = "Stat Fcst "
        all_stat_forecast_cols = [
            c for c in ForecastData_pl.columns if c.startswith(stat_fcst_str) and "L1" not in c
        ]
        forecast_columns = [col.removesuffix(" Planning Cycle") for col in all_stat_forecast_cols]
        validation_error_col = "Validation Error"
        composite_error_col = "Composite Error"
        sell_out_offset_col = "Offset Period"

        selection_criteria_str = SelectionCriteria_pl.get_column(
            o9Constants.BESTFIT_SELECTION_CRITERIA
        )[0]

        best_fit_cols = (
            [o9Constants.VERSION_NAME] + forecast_level + [o9Constants.SYSTEM_BESTFIT_ALGORITHM]
        )
        validation_cols = (
            [o9Constants.VERSION_NAME]
            + forecast_level
            + [
                o9Constants.STAT_RULE,
                o9Constants.STAT_ALGORITHM,
                validation_error_col,
                composite_error_col,
                "Validation Actual",
                "Validation Fcst",
                "Validation Fcst Abs Error",
            ]
        )
        ValidationErrorPlanningCycle_cols = (
            [o9Constants.VERSION_NAME, o9Constants.PLANNING_CYCLE_DATE]
            + forecast_level
            + [o9Constants.STAT_ALGORITHM, "Validation Error Planning Cycle"]
        )
        validationforecast_cols = (
            [o9Constants.VERSION_NAME]
            + forecast_level
            + [o9Constants.PARTIAL_WEEK]
            + forecast_columns
        )

        if ForecastEngine_pl is not None and not ForecastEngine_pl.is_empty():
            if ForecastEngine_pl.get_column(o9Constants.FORECAST_ENGINE)[0] == "ML":
                logger.info("Selected engine is ML, returning BestfitAlgo as CML.")
                best_fit_cols = [o9Constants.VERSION_NAME] + forecast_level
                BestFitAlgo = AssignedAlgoList_pl
                BestFitAlgo = BestFitAlgo.with_columns(
                    pl.lit("CML").alias(o9Constants.SYSTEM_BESTFIT_ALGORITHM)
                )
                return (
                    BestFitAlgo,
                    pl.DataFrame(schema=validation_cols),
                    pl.DataFrame(schema=ValidationErrorPlanningCycle_cols),
                    pl.DataFrame(schema=validationforecast_cols),
                )

        time_configs = {
            "Week": {
                "name": "Time.[Week]",
                "key": "Time.[WeekKey]",
                "join_col": "Time.[Partial Week]",
            },
            "Planning Month": {
                "name": "Time.[Planning Month]",
                "key": "Time.[PlanningMonthKey]",
                "join_col": "Time.[Partial Week]",
            },
            "Month": {
                "name": "Time.[Month]",
                "key": "Time.[MonthKey]",
                "join_col": "Time.[Partial Week]",
            },
            "Planning Quarter": {
                "name": "Time.[Planning Quarter]",
                "key": "Time.[PlanningQuarterKey]",
                "join_col": "Time.[Partial Week]",
            },
            "Quarter": {
                "name": "Time.[Quarter]",
                "key": "Time.[QuarterKey]",
                "join_col": "Time.[Partial Week]",
            },
        }

        ForecastGenTimeBucket_str = ForecastGenTimeBucket_pl.get_column(
            o9Constants.FORECAST_GEN_TIME_BUCKET
        )[0]
        time_config = time_configs[ForecastGenTimeBucket_str]

        error_metric = ForecastParameters_pl.get_column("Error Metric")[0]
        override_flat_line = eval(OverrideFlatLineForecasts_str)

        if SellOutOffset_pl is None or SellOutOffset_pl.height == 0:
            SellOutOffset_pl = pl.DataFrame(
                {
                    o9Constants.VERSION_NAME: [
                        ForecastGenTimeBucket_pl.get_column(o9Constants.VERSION_NAME)[0]
                    ],
                    sell_out_offset_col: [0],
                }
            )

        offset_period = int(SellOutOffset_pl.select(sell_out_offset_col).item())

        # Calculation of Valid Planning Cycles
        planning_cycle_dates = calculate_planning_cycle_dates(
            validation_params_df=ValidationParameters_pl,
            PlanningCycleDates_pl=PlanningCycleDates_pl,
            current_time_period_df=CurrentTimePeriod_pl,
            time_dimension_df=TimeDimension_pl,
            time_name_col=time_config["name"],
            time_key_col=time_config["key"],
            offset_period=offset_period,
        )

        # getting latest planning cycle date
        latest_planning_cycle_date = planning_cycle_dates[0]

        # Filtering out the ForecastData for Valid Planning Cycles
        ForecastData_pl = ForecastData_pl.filter(
            pl.col(o9Constants.PLANNING_CYCLE_DATE).is_in(planning_cycle_dates)
        )

        # Forecast data for latest planning cycle
        ValidationForecast = (
            ForecastData_pl.filter(
                pl.col(o9Constants.PLANNING_CYCLE_DATE) == latest_planning_cycle_date
            )
            .drop(o9Constants.PLANNING_CYCLE_DATE)
            .rename({col: col.removesuffix(" Planning Cycle") for col in all_stat_forecast_cols})
        )

        # Merging Actuals and ForecastData on Forecast Level
        ActualsandForecastData_pl = Actuals_pl.join(
            ForecastData_pl, on=forecast_level + [o9Constants.PARTIAL_WEEK], how="inner"
        )

        # Create Integer ID Mapping for Grains
        base_data = _prepare_base_data(
            ActualsandForecastData_pl,
            TimeDimension_pl.select(
                [time_config["join_col"], time_config["name"], time_config["key"]]
            ).unique(),
            forecast_level,
            HistoryMeasure_str,
            time_config,
            all_stat_forecast_cols,
        )

        # Create the mapping table from grain columns to a unique integer ID
        id_mapping_df = (
            base_data.select(forecast_level).unique().with_row_count("unique_id", offset=1)
        )

        logger.info(f"Base DataFrame shape before ID Mapping: {id_mapping_df.shape}")

        # Join the integer ID and drop the original grain columns for leaner processing
        processing_data = base_data.join(id_mapping_df, on=forecast_level).drop(forecast_level)

        logger.info(f"Processing DataFrame shape after ID Mapping: {processing_data.shape}")

        relevant_time_mapping = TimeDimension_pl.select(
            [time_config["name"], time_config["key"]]
        ).unique()
        latest_time_name = get_last_time_period(
            CurrentTimePeriod_pl, relevant_time_mapping, time_config["name"], time_config["key"]
        )

        if not latest_time_name:
            logger.warning("Could not determine the last time period. Aborting.")
            return (
                pl.DataFrame(schema=best_fit_cols),
                pl.DataFrame(schema=validation_cols),
                pl.DataFrame(schema=ValidationErrorPlanningCycle_cols),
                ValidationForecast.select(validationforecast_cols),
            )

        AllPlanningPeriods = (
            ValidationParameters_pl["Validation Period"]
            + (ValidationParameters_pl["Validation Fold"] - 1)
            * ValidationParameters_pl["Validation Step Size"]
        )

        validation_period_dates = get_n_time_periods(
            latest_time_name,
            -int(AllPlanningPeriods[0] + offset_period),
            relevant_time_mapping,
            time_config["name"],
            time_config["key"],
            True,
        )[: -offset_period or None]

        validation_data = (
            processing_data.filter(pl.col(time_config["name"]).is_in(validation_period_dates))
            .join(relevant_time_mapping, on=time_config["name"])
            .sort(time_config["key"])
            .drop(time_config["key"])
            .with_columns(pl.col(HistoryMeasure_str).clip(lower_bound=0).fill_null(0))
        )
        if validation_data.is_empty():
            logger.warning("No data available for the validation period. Aborting.")
            return (
                pl.DataFrame(schema=best_fit_cols),
                pl.DataFrame(schema=validation_cols),
                pl.DataFrame(schema=ValidationErrorPlanningCycle_cols),
                ValidationForecast.select(validationforecast_cols),
            )

        logger.info("Calculating Validation error")

        # Calculate Errors and Select Best Fit
        agg_cals, validation_error_planningcycle = _calculate_validation_error(
            validation_data,
            HistoryMeasure_str,
            validation_error_col,
            stat_fcst_str,
            error_metric,
            override_flat_line,
        )

        if validation_error_planningcycle.is_empty():
            return (
                pl.DataFrame(schema=best_fit_cols),
                pl.DataFrame(schema=validation_cols),
                pl.DataFrame(schema=ValidationErrorPlanningCycle_cols),
                ValidationForecast.select(validationforecast_cols),
            )

        # Filtering out the AssingedAlgoList Logic
        AssignedAlgoList_pl = AssignedAlgoList_pl.join(id_mapping_df, on=forecast_level).drop(
            forecast_level
        )

        filter_df = (
            AssignedAlgoList_pl.select(
                [o9Constants.ASSIGNED_ALGORITHM_LIST, "unique_id", o9Constants.ASSIGNED_RULE]
            )
            .with_columns(
                (
                    pl.col(o9Constants.ASSIGNED_ALGORITHM_LIST)
                    .str.split(by=",")
                    .list.concat(pl.lit(["CML"]))
                    if (validation_data[stat_fcst_str + "CML Planning Cycle"].is_null().sum() == 0)
                    else pl.col(o9Constants.ASSIGNED_ALGORITHM_LIST).str.split(by=",")
                ).alias(o9Constants.ASSIGNED_ALGORITHM_LIST)
            )
            .explode(o9Constants.ASSIGNED_ALGORITHM_LIST)
            .rename(
                {
                    o9Constants.ASSIGNED_ALGORITHM_LIST: o9Constants.STAT_ALGORITHM,
                    o9Constants.ASSIGNED_RULE: o9Constants.STAT_RULE,
                }
            )
        )

        validation_error_planningcycle = validation_error_planningcycle.join(
            filter_df, on=["unique_id", o9Constants.STAT_ALGORITHM], how="inner"
        ).rename({validation_error_col: "Validation Error Planning Cycle"})

        algos_to_drop = (
            validation_error_planningcycle.filter(
                pl.col("Validation Error Planning Cycle").is_null()
            )
            .select(["unique_id", o9Constants.STAT_ALGORITHM])
            .unique()
        )

        logger.info(f"Algorithms to drop for each intersection{algos_to_drop}")

        validation_error_df = (
            validation_error_planningcycle.join(
                algos_to_drop, on=["unique_id", o9Constants.STAT_ALGORITHM], how="anti"
            )
            .group_by(["unique_id", o9Constants.STAT_ALGORITHM, o9Constants.STAT_RULE])
            .agg(pl.mean("Validation Error Planning Cycle").alias(validation_error_col))
            .sort(["unique_id", o9Constants.STAT_ALGORITHM])
        )

        validation_error_planningcycle = validation_error_planningcycle.drop_nulls()

        validation_error_df = validation_error_df.join(
            agg_cals, on=["unique_id", o9Constants.STAT_ALGORITHM], how="left"
        )

        if (
            selection_criteria_str == "Validation Error plus Violations"
            and not Violations_pl.is_empty()
            and not Weights_pl.is_empty()
        ):
            logger.info("Calculating composite error with violations...")

            # Create a config dictionary for all the column names
            composite_config = {
                "error_metric": error_metric,
                "val_error_col": validation_error_col,
                "composite_error_col": composite_error_col,
                "val_actual_col": "Validation Actual",
                "val_abs_error_col": "Validation Fcst Abs Error",
                "weights_join_cols": ["Item.[Segmentation LOB]"],
                "composite_join_cols": [
                    "unique_id",
                    o9Constants.STAT_ALGORITHM,
                ],
                "weighted_cols": [
                    "unique_id",
                    o9Constants.STAT_ALGORITHM,
                    o9Constants.STAT_RULE,
                ],
                "violation_cols": [
                    "Straight Line",
                    "Trend Violation",
                    "Seasonal Violation",
                    "Level Violation",
                    "Range Violation",
                    "COCC Violation",
                ],
                "weight_cols": [
                    "Straight Line Weight",
                    "Trend Weight",
                    "Seasonality Weight",
                    "Level Weight",
                    "Range Weight",
                    "COCC Weight",
                ],
            }

            # Call the new function. Note that final_error_df already has the required columns.
            final_error_df = _calculate_composite_error(
                df_error=validation_error_df,
                df_violations=Violations_pl,
                df_weights=Weights_pl,
                id_mapping_df=id_mapping_df,
                forecast_level=forecast_level,
                config=composite_config,
            )
        else:
            # If not calculating composite error, just alias the validation error column
            final_error_df = validation_error_df.with_columns(
                pl.col(validation_error_col).alias(composite_error_col)
            )

        master_algo_list = get_algo_ranking()

        BestFitAlgo_pl = _select_best_fit(
            final_error_df, "unique_id", composite_error_col, master_algo_list
        )

        # Join the forecast_level columns back using the mapping table
        BestFitAlgo_pl = BestFitAlgo_pl.join(id_mapping_df, on="unique_id", how="left").drop(
            "unique_id"
        )
        ValidationError_pl = final_error_df.join(id_mapping_df, on="unique_id", how="left").drop(
            "unique_id"
        )
        ValidationErrorPlanningCycle_pl = validation_error_planningcycle.join(
            id_mapping_df, on="unique_id", how="left"
        ).drop("unique_id")
        input_version = Actuals_pl.get_column(o9Constants.VERSION_NAME)[0]
        BestFitAlgo_pl = BestFitAlgo_pl.with_columns(
            pl.lit(input_version).alias(o9Constants.VERSION_NAME)
        )
        ValidationError_pl = ValidationError_pl.with_columns(
            pl.lit(input_version).alias(o9Constants.VERSION_NAME)
        )
        ValidationErrorPlanningCycle_pl = ValidationErrorPlanningCycle_pl.with_columns(
            pl.lit(input_version).alias(o9Constants.VERSION_NAME)
        )

        return (
            BestFitAlgo_pl[best_fit_cols],
            ValidationError_pl[validation_cols],
            ValidationErrorPlanningCycle_pl[ValidationErrorPlanningCycle_cols],
            ValidationForecast[validationforecast_cols],
        )

    except Exception as e:
        logger.exception(f"A critical error occurred before starting iterations for slice : {e}")
        return (
            pl.DataFrame(schema=best_fit_cols),
            pl.DataFrame(schema=validation_cols),
            pl.DataFrame(schema=ValidationErrorPlanningCycle_cols),
            pl.DataFrame(schema=validationforecast_cols),
        )


col_mapping = {
    "Validation Error": float,
    "Validation Error Planning Cycle": float,
    "Composite Error": float,
    "System Bestfit Algorithm": str,
    "Validation Actual": float,
    "Validation Fcst": float,
    "Validation Fcst Abs Error": float,
    "Stat Fcst SES": float,
    "Stat Fcst DES": float,
    "Stat Fcst TES": float,
    "Stat Fcst ETS": float,
    "Stat Fcst Auto ARIMA": float,
    "Stat Fcst sARIMA": float,
    "Stat Fcst Prophet": float,
    "Stat Fcst STLF": float,
    "Stat Fcst Theta": float,
    "Stat Fcst Croston": float,
    "Stat Fcst TBATS": float,
    "Stat Fcst AR-NNET": float,
    "Stat Fcst Simple Snaive": float,
    "Stat Fcst Weighted Snaive": float,
    "Stat Fcst Growth Snaive": float,
    "Stat Fcst Naive Random Walk": float,
    "Stat Fcst Seasonal Naive YoY": float,
    "Stat Fcst Moving Average": float,
    "Stat Fcst Simple AOA": float,
    "Stat Fcst Growth AOA": float,
    "Stat Fcst Weighted AOA": float,
    "Stat Fcst SCHM": float,
    "Stat Fcst CML": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    HistoryMeasure,
    TimeDimension,
    ForecastParameters,
    CurrentTimePeriod,
    Actuals,
    ForecastData,
    ValidationParameters,
    OverrideFlatLineForecasts,
    ForecastGenTimeBucket,
    AssignedAlgoList,
    SelectionCriteria,
    MasterAlgoList,
    PlanningCycleDates,
    df_keys,
    Weights=None,
    Violations=None,
    ForecastEngine=None,
    SellOutOffset=None,
    multiprocessing_num_cores=1,
):
    """
    Main entry point for the plugin.
    This function handles pandas-to-polars conversion, iterates through forecast
    cycles, calls the core Polars pipeline, and converts results back to pandas.
    """
    (
        BestFitAlgoList,
        ValidationErrorList,
        ValidationErrorPlanningCycleList,
        ValidationForecastList,
    ) = ([], [], [], [])
    try:

        base_inputs = {
            "Grains_str": Grains,
            "HistoryMeasure_str": HistoryMeasure,
            "OverrideFlatLineForecasts_str": OverrideFlatLineForecasts,
            "ForecastGenTimeBucket_pl": pl.from_pandas(ForecastGenTimeBucket),
            "SelectionCriteria_pl": pl.from_pandas(SelectionCriteria),
            "TimeDimension_pl": pl.from_pandas(TimeDimension),
            "PlanningCycleDates_pl": pl.from_pandas(PlanningCycleDates),
            "ForecastParameters_pl": pl.from_pandas(ForecastParameters),
            "CurrentTimePeriod_pl": pl.from_pandas(CurrentTimePeriod),
            "Actuals_pl": pl.from_pandas(Actuals),
            "ForecastData_pl": pl.from_pandas(ForecastData),
            "ValidationParameters_pl": pl.from_pandas(ValidationParameters),
            "AssignedAlgoList_pl": pl.from_pandas(AssignedAlgoList),
            "MasterAlgoList_pl": pl.from_pandas(MasterAlgoList),
            "Weights_pl": pl.from_pandas(Weights) if Weights is not None else None,
            "Violations_pl": pl.from_pandas(Violations) if Violations is not None else None,
            "SellOutOffset_pl": (
                pl.from_pandas(SellOutOffset) if SellOutOffset is not None else None
            ),
            "ForecastEngine_pl": (
                pl.from_pandas(ForecastEngine) if ForecastEngine is not None else None
            ),
        }

        for iter_val in ForecastGenTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            try:
                iter_inputs = build_iter_inputs_from_dict(base_inputs, iter_val)
                (
                    the_best_fit_algo_pl,
                    the_validation_error_pl,
                    the_validation_error_planningcycle_pl,
                    the_validation_forecast_pl,
                ) = run_best_fit_analysis_pl(**iter_inputs)

                logger.info(f"--- Processing iteration: {iter_val} for slice: {df_keys} ---")

                BestFitAlgoList.append(
                    the_best_fit_algo_pl.select(
                        pl.lit(iter_val).alias(o9Constants.FORECAST_ITERATION), pl.all()
                    ).to_pandas()
                )
                ValidationErrorList.append(
                    the_validation_error_pl.select(
                        pl.lit(iter_val).alias(o9Constants.FORECAST_ITERATION), pl.all()
                    ).to_pandas()
                )
                ValidationErrorPlanningCycleList.append(
                    the_validation_error_planningcycle_pl.select(
                        pl.lit(iter_val).alias(o9Constants.FORECAST_ITERATION), pl.all()
                    ).to_pandas()
                )
                ValidationForecastList.append(
                    the_validation_forecast_pl.select(
                        pl.lit(iter_val).alias(o9Constants.FORECAST_ITERATION), pl.all()
                    ).to_pandas()
                )

            except Exception as e:
                logger.exception(
                    f"Failed to process iteration '{iter_val}' for slice {df_keys}: {e}. Skipping."
                )
                continue

        BestFitAlgo = (
            pd.concat(BestFitAlgoList, ignore_index=True) if BestFitAlgoList else pd.DataFrame()
        )
        ValidationError = (
            pd.concat(ValidationErrorList, ignore_index=True)
            if ValidationErrorList
            else pd.DataFrame()
        )
        ValidationErrorPlanningCycle = (
            pd.concat(ValidationErrorPlanningCycleList, ignore_index=True)
            if ValidationErrorPlanningCycleList
            else pd.DataFrame()
        )
        ValidationForecast = (
            pd.concat(ValidationForecastList, ignore_index=True)
            if ValidationForecastList
            else pd.DataFrame()
        )

    except Exception as e:
        logger.exception(
            f"A critical error occurred before starting iterations for slice {df_keys}: {e}"
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return BestFitAlgo, ValidationError, ValidationErrorPlanningCycle, ValidationForecast
