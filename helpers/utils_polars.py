import logging
from functools import wraps
from typing import Any, Dict, List

import pandas as pd
import polars as pl

from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")


def create_algo_list(AlgoDF: pl.DataFrame, algo_list_col: str, use_all_algos: bool = False) -> list:
    if use_all_algos:
        return get_algo_ranking()

    if AlgoDF.is_empty():
        logger.warning("No Algorithms were assigned...")
        logger.warning("Returning empty list...")
        return []

    # Extract the first value from the specified column
    algos = AlgoDF.select(algo_list_col).item()

    # Split by comma and clean whitespace
    algo_list = [item.strip() for item in algos.split(",")]

    # Return deduplicated list
    return list(set(algo_list))


def get_algo_ranking() -> list:
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
    ]
    return ordered_list


def insert_forecast_iteration(df: pd.DataFrame, the_iteration: str) -> pd.DataFrame:
    if df.empty and o9Constants.FORECAST_ITERATION not in df.columns:
        new_column_list = [o9Constants.FORECAST_ITERATION] + list(df.columns)
        df = pd.DataFrame(columns=new_column_list)
    else:
        if o9Constants.FORECAST_ITERATION not in df.columns:
            df.insert(
                loc=0,
                column=o9Constants.FORECAST_ITERATION,
                value=the_iteration,
            )
    return df


def filter_for_iteration_polars(iteration: str):
    def decorator(func):
        @wraps(func)
        def wrapper(**kwargs):
            logger.info(f"Filtering input dataframes for iteration {iteration}")

            def _drop_extra_cols(df: pl.DataFrame) -> pl.DataFrame:
                if df is None or df.is_empty():
                    return df
                drop_cols = []
                # Always consider these two
                if o9Constants.FORECAST_ITERATION in df.columns:
                    drop_cols.append(o9Constants.FORECAST_ITERATION)
                if o9Constants.SEQUENCE in df.columns:
                    drop_cols.append(o9Constants.SEQUENCE)
                # Drop any column that contains the substring
                drop_cols.extend([c for c in df.columns if "Slice Association" in c])
                return df.drop(drop_cols) if drop_cols else df

            filtered_args = dict()
            for the_arg, the_value in kwargs.items():
                if the_arg == "StatFcstL1ForFIPLIteration":
                    filtered_args[the_arg] = the_value
                    continue

                if isinstance(the_value, pl.DataFrame):
                    if o9Constants.FORECAST_ITERATION in the_value.columns:
                        filtered_df = the_value.filter(
                            pl.col(o9Constants.FORECAST_ITERATION) == iteration
                        )
                        filtered_args[the_arg] = _drop_extra_cols(filtered_df)

                    elif o9Constants.FORECAST_ITERATION_SELECTION in the_value.columns:
                        filtered_df = the_value.filter(
                            pl.col(o9Constants.FORECAST_ITERATION_SELECTION).is_not_null()
                        ).filter(
                            pl.col(o9Constants.FORECAST_ITERATION_SELECTION).str.contains(iteration)
                        )
                        filtered_args[the_arg] = _drop_extra_cols(filtered_df)
                    else:
                        # No iteration column; still drop extra columns if present
                        filtered_args[the_arg] = _drop_extra_cols(the_value)
                else:
                    filtered_args[the_arg] = the_value

            result = func(**filtered_args)

            logger.info(f"Adding {o9Constants.FORECAST_ITERATION} : {iteration} to output ...")

            if isinstance(result, pl.DataFrame):
                result = insert_forecast_iteration(result, iteration)
            elif isinstance(result, tuple):
                result = tuple(insert_forecast_iteration(x, iteration) for x in result)

            return result

        return wrapper

    return decorator


def get_planning_to_stat_region_mapping(
    RegionLevel: pl.DataFrame,
    RegionMasterData: pl.DataFrame,
) -> pl.DataFrame:
    # Extract region level value
    region_level = RegionLevel.select(o9Constants.REGION_LEVEL).item()
    logger.info(f"{o9Constants.REGION_LEVEL} : {region_level}")

    required_col = f"Region.[{region_level}]"
    logger.debug(f"required_col : {required_col}")

    if required_col == o9Constants.PLANNING_REGION:
        # Use only the planning region column, make a copy with renamed stat region
        RegionMasterData = (
            RegionMasterData.select([o9Constants.PLANNING_REGION])
            .unique()
            .with_columns([pl.col(o9Constants.PLANNING_REGION).alias(o9Constants.STAT_REGION)])
        )
    else:
        # Select planning region and required column, rename required_col to stat region
        RegionMasterData = (
            RegionMasterData.select([o9Constants.PLANNING_REGION, required_col])
            .unique()
            .rename({required_col: o9Constants.STAT_REGION})
        )

    return RegionMasterData


def get_last_time_period(
    current_time_df: pl.DataFrame,
    time_dimension: pl.DataFrame,
    time_column: str,
    key_column: str,
) -> str:
    """Polars implementation to get the last time period before the current one."""
    if current_time_df.is_empty() or time_dimension.is_empty():
        return ""

    current_time_key = current_time_df.get_column(key_column)[0]

    last_period_df = (
        time_dimension.select(time_column, key_column)
        .unique()
        .filter(pl.col(key_column) < current_time_key)
        .sort(key_column)
        .tail(1)
    )

    if last_period_df.is_empty():
        logger.error("No previous time period exists, returning empty string.")
        return ""

    return last_period_df.get_column(time_column)[0]


def get_n_time_periods(
    latest_value: str,
    periods: int,
    time_mapping: pl.DataFrame,
    time_name_col: str,
    time_key_col: str,
    include_latest_value: bool = True,
) -> List[str]:
    """Polars implementation to get n time periods based on values provided."""
    req_time_mapping = (
        time_mapping.select(time_name_col, time_key_col)
        .unique()
        .with_columns(pl.col(time_key_col).cast(pl.Datetime))
        .sort(time_key_col)
    )
    try:
        index_of_latest_value = (
            req_time_mapping.get_column(time_name_col).to_list().index(latest_value)
        )
    except ValueError:
        logger.error(f"latest_value '{latest_value}' not found in time dimension.")
        return []
    if periods > 0:
        start_index = index_of_latest_value if include_latest_value else index_of_latest_value + 1
        end_index = start_index + periods
        result = req_time_mapping[start_index:end_index]
    else:
        end_index = index_of_latest_value + 1 if include_latest_value else index_of_latest_value
        start_index = max(0, end_index + periods)
        result = req_time_mapping[start_index:end_index]
    return result.get_column(time_name_col).to_list()


def get_default_algo_params_nixtla(
    stat_algo_col: str,
    stat_parameter_col: str,
    system_stat_param_value_col: str,
    frequency: str,
    DefaultAlgoParameters: pl.DataFrame,
    monthly_default_col: str = "Stat Parameter.[Stat Parameter Monthly Default]",
    weekly_default_col: str = "Stat Parameter.[Stat Parameter Weekly Default]",
) -> pl.DataFrame:

    freq_to_col = {
        "Weekly": "Stat Parameter.[Stat Parameter Weekly Default]",
        "Monthly": "Stat Parameter.[Stat Parameter Monthly Default]",
        "Planning Month": "Stat Parameter.[Stat Parameter Monthly Default]",
    }
    default_col = freq_to_col.get(frequency)

    if default_col is None:
        raise ValueError(f"Unsupported frequency: {frequency}")
    if not isinstance(default_col, str):
        raise TypeError(
            f"default_col should be a column name (str), got {type(default_col)}: {default_col}"
        )

    missing = [
        col
        for col in [stat_algo_col, stat_parameter_col, default_col]
        if col not in DefaultAlgoParameters.columns
    ]
    if missing:
        raise ValueError(f"Missing columns in DefaultAlgoParameters: {missing}")

    return (
        DefaultAlgoParameters.select([stat_algo_col, stat_parameter_col, default_col])
        .unique()
        .rename({default_col: system_stat_param_value_col})
    )


def get_default_algo_params_polars(
    stat_algo_col,
    stat_parameter_col,
    system_stat_param_value_col,
    frequency,
    intersections_master: pl.DataFrame,
    DefaultAlgoParameters: pl.DataFrame,
    quarterly_default_col="Stat Parameter.[Stat Parameter Quarterly Default]",
    monthly_default_col="Stat Parameter.[Stat Parameter Monthly Default]",
    weekly_default_col="Stat Parameter.[Stat Parameter Weekly Default]",
):
    # Assert columns exist
    for col in [quarterly_default_col, monthly_default_col, weekly_default_col]:
        assert col in DefaultAlgoParameters.columns, f"Missing column: {col}"

    # Default to monthly
    col_to_select = monthly_default_col

    if frequency == "Weekly":
        col_to_select = weekly_default_col
    elif frequency == "Quarterly":
        col_to_select = quarterly_default_col

    cols_to_select = [stat_algo_col, stat_parameter_col, col_to_select]

    # Select, unique, rename
    AlgoParameters_df = (
        DefaultAlgoParameters.select(cols_to_select)
        .unique()
        .rename({col_to_select: system_stat_param_value_col})
    )

    # Polars cross join:
    AlgoParameters_for_all_intersections = AlgoParameters_df.join(intersections_master, how="cross")
    return AlgoParameters_for_all_intersections


def get_holidays_at_stat_region_level(
    HolidayData: pl.DataFrame,
    StatRegionMapping: pl.DataFrame,
    planning_region_col: str,
    stat_region_col: str,
    time_day_col: str,
    holiday_type_col: str,
    TimeDimension: pl.DataFrame,
    relevant_time_name: str,
) -> pl.DataFrame:

    if HolidayData.is_empty():
        return pl.DataFrame()

    # Join with StatRegionMapping to map planning region to stat region
    HolidayData = (
        HolidayData.join(StatRegionMapping, on=planning_region_col, how="inner")
        .select([stat_region_col, time_day_col, holiday_type_col])
        .unique()
        .join(
            TimeDimension.select([time_day_col, relevant_time_name]), on=time_day_col, how="inner"
        )
        .groupby([stat_region_col, relevant_time_name])
        .agg(
            [
                pl.col(holiday_type_col)
                .implode()
                .map_elements(lambda x: ",".join(x))
                .alias(holiday_type_col)
            ]
        )
        .select([stat_region_col, relevant_time_name, holiday_type_col])
    )

    return HolidayData


def extract_algo_params_batch(
    df: pl.DataFrame, version: str, granularity: str = "Weekly", seasonal_periods: int = 52
) -> dict:
    """
    Generic extraction of all stat model parameters per algorithm in batch mode using Polars.
    Returns: Dict of algorithm -> param dict (all params found for that algo)
    """
    value_col = f"Stat Parameter.[Stat Parameter {granularity} Default]"
    algo_col = "Stat Algorithm.[Stat Algorithm]"
    param_col = "Stat Parameter.[Stat Parameter]"
    system_val_col = "System Stat Parameter Value"

    # Filter by version
    df = df.filter(pl.col("Version.[Version Name]") == version)

    algo_param_dict = {}

    for algo in df[algo_col].unique().to_list():
        algo_df = df.filter(pl.col(algo_col) == algo)
        params = {}

        for param in algo_df[param_col].unique().to_list():
            param_df = algo_df.filter(pl.col(param_col) == param)
            val = None

            # Check for value_col first, then system_val_col
            value = param_df.filter(~pl.col(value_col).is_null())
            if value.height > 0:
                val = value.select(pl.col(value_col)).to_series()[0]
            else:
                system_val = param_df.filter(~pl.col(system_val_col).is_null())
                if system_val.height > 0:
                    val = system_val.select(pl.col(system_val_col)).to_series()[0]
                else:
                    continue

            # Handle string/float logic
            if param == "Growth Type":
                params[param] = str(val)
            else:
                try:
                    params[param] = round(float(val), 2)
                except Exception:
                    params[param] = val
        if params:
            algo_param_dict[algo] = params

    return algo_param_dict


def disaggregate_data(
    source_df: pl.DataFrame,
    source_grain: str,
    target_grain: str,
    profile_df: pl.DataFrame,
    profile_col: str,
    cols_to_disaggregate: list,
    logger=None,  # Optional: Pass a logger if you want
) -> pl.DataFrame:
    """
    Function to disaggregate data from higher to lower level based on ratios, using Polars Lazy API.
    """
    if logger:
        logger.info(f"Disaggregating values from {source_grain} to {target_grain}")

    # Check if inputs are empty
    if source_df.height == 0 or profile_df.height == 0:
        if logger:
            logger.warning("source_df/profile_df is empty, returning empty dataframe")
        return pl.DataFrame()

    # Column existence checks
    for col in [source_grain, *cols_to_disaggregate]:
        assert col in source_df.columns, f"{col} not found in source_df"
    for col in [source_grain, target_grain, profile_col]:
        assert col in profile_df.columns, f"{col} not found in profile_df"

    try:
        if logger:
            logger.info(f"source_df shape: {source_df.shape}, profile_df shape: {profile_df.shape}")

        # Convert to lazy
        source_lazy = source_df.lazy()
        profile_lazy = profile_df.lazy()

        # Inner join on source_grain
        combined_lazy = source_lazy.join(profile_lazy, on=source_grain, how="inner")

        # Disaggregate specified columns
        disagg_exprs = []
        disagg_cols = []

        for the_col in cols_to_disaggregate:
            if the_col in source_df.columns:
                backup_col_name = the_col + "_aggregated"
                disagg_cols.append(the_col)
                # Rename and multiply in chain using alias
                disagg_exprs.append((pl.col(the_col) * pl.col(profile_col)).alias(the_col))
                # Keep original as backup_col
                disagg_exprs.append(pl.col(the_col).alias(backup_col_name))
            else:
                if logger:
                    logger.warning(f"{the_col} not found in DataFrame, skipping...")

        # Build final columns list: keep all original columns except source_grain and disagg_cols, add target_grain, disagg_cols
        req_cols = (
            [col for col in source_df.columns if col not in [source_grain] + disagg_cols]
            + [target_grain]
            + disagg_cols
        )

        # Chain the expressions (apply all)
        disaggregated_lazy = (
            combined_lazy.with_columns(disagg_exprs).select(req_cols).unique()  # drop_duplicates
        )

        disaggregated_df = disaggregated_lazy.collect()

        if logger:
            logger.info(
                f"Disaggregation complete, df shape: {disaggregated_df.shape}, columns: {disaggregated_df.columns}"
            )

    except Exception as e:
        if logger:
            logger.exception(e)
        return pl.DataFrame()

    return disaggregated_df


def convert_dict_to_polars(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (pl.from_pandas(v) if isinstance(v, pd.DataFrame) else v) for k, v in inputs.items()}


def filter_inputs_dict_by_iteration(inputs: Dict[str, Any], iteration: str) -> Dict[str, Any]:
    @filter_for_iteration_polars(iteration)
    def _identity(**kwargs):
        return kwargs

    return _identity(**inputs)


def build_iter_inputs_from_dict(
    pandas_or_polars_inputs: Dict[str, Any],
    iteration: str,
) -> Dict[str, Any]:
    pl_inputs = convert_dict_to_polars(pandas_or_polars_inputs)
    return filter_inputs_dict_by_iteration(pl_inputs, iteration)
