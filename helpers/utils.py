import logging

import pandas as pd
import polars as pl
import pyspark
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    create_cartesian_product,
)
from o9Reference.spark_utils.common_utils import get_clean_string, is_dimension
from pyspark.sql.types import StructType

from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")


def get_measure_name(key: str):
    return "Stat Fcst " + key


def get_model_desc_name(key: str):
    return key + " Model"


def get_bound_col_name(measure_name, alpha, direction):
    # assume alpha = 0.05, we want to publish 95 into interval
    interval = int(100 - (alpha * 100))
    return measure_name + " {}% {}".format(interval, direction)


def get_ts_freq_prophet(seasonal_periods):
    # Assign default value of 12
    ts_freq = "M"  # Month Start
    if seasonal_periods == 52 or seasonal_periods == 53:
        ts_freq = "W"  # Weekly
    elif seasonal_periods == 4:
        ts_freq = "Q"  # Quarter Start
    return ts_freq


def get_seasonal_periods(frequency: str) -> int:
    """
    Returns num seasonal periods based on string
    """
    if frequency == "Weekly":
        return 52
    elif frequency == "Monthly":
        return 12
    elif frequency == "Quarterly":
        return 4
    else:
        raise ValueError("Unknown frequency {}".format(frequency))


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


def filter_for_iteration(iteration: str):
    def decorator(func):
        def wrapper(**kwargs):
            logger.info(f"Filtering input dataframes for iteration {iteration}")

            # Filter dataframes if they are provided
            filtered_args = dict()
            for the_arg, the_value in kwargs.items():

                # exception : DP016TransitionLevelStat - StatFcstL1ForFIPLIteration - this will have data for FI-PL iteration but scope for the plugin would be some other iteration
                # do not filter this input dataframe for iteration
                if the_arg == "StatFcstL1ForFIPLIteration":
                    filtered_args[the_arg] = the_value
                    continue

                if isinstance(the_value, pd.DataFrame):
                    if o9Constants.FORECAST_ITERATION in the_value.columns:
                        # filter data for the iteration
                        filtered_df = the_value[
                            the_value[o9Constants.FORECAST_ITERATION] == iteration
                        ].copy(deep=True)

                        # remove iteration column to avoid join related issues inside the plugin
                        filtered_df.drop(
                            o9Constants.FORECAST_ITERATION,
                            axis=1,
                            inplace=True,
                        )

                        # add it back with the same key
                        filtered_args[the_arg] = filtered_df

                    elif o9Constants.FORECAST_ITERATION_SELECTION in the_value.columns:
                        # remove nulls if present
                        the_value = the_value[
                            the_value[o9Constants.FORECAST_ITERATION_SELECTION].notna()
                        ]

                        # filter data for the iteration
                        filtered_df = the_value[
                            the_value[o9Constants.FORECAST_ITERATION_SELECTION].str.contains(
                                iteration
                            )
                        ].copy(deep=True)

                        # add it back with the same key
                        filtered_args[the_arg] = filtered_df

                    else:
                        filtered_args[the_arg] = the_value
                else:
                    filtered_args[the_arg] = the_value

            # call the function with the updated arguments
            result = func(**filtered_args)

            logger.info(f"Adding {o9Constants.FORECAST_ITERATION} : {iteration} to output ...")

            if isinstance(result, pd.DataFrame):
                # check and insert forecast iteration column
                result = insert_forecast_iteration(df=result, the_iteration=iteration)
            elif isinstance(result, tuple):
                # check and insert forecast iteration column for every dataframe
                result = [insert_forecast_iteration(df=x, the_iteration=iteration) for x in result]

            return result

        return wrapper

    return decorator


def get_list_of_grains():
    # TODO : Implement function which checks dtype and can work with name nodes
    return


def get_list_of_grains_from_string(input: str, delimiter: str = ",") -> list:
    if input == "None" or input is None:
        return list()

    # split on delimiter and obtain grains
    all_grains = input.split(delimiter)

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get segmentation level
    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    return dimensions


# Function to assign values based on conditions
def get_rule(row: pd.Series) -> str:
    try:
        if row[o9Constants.STAT_ALGORITHM] in row[o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST]:
            return row[o9Constants.ASSIGNED_RULE]
        # # case where both supersets are not available, return assigned rule itself
        # elif math.isnan(
        #     row[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST]
        # ) or math.isnan(row[o9Constants.PLANNER_BESTFIT_ALGORITHM]):
        #     return row[o9Constants.ASSIGNED_RULE]
        elif (
            row[o9Constants.STAT_ALGORITHM] == "Ensemble"
            and o9Constants.SYSTEM_ENSEMBLE_ALGORITHM_LIST in row.index
        ):
            system_ensemble_algo = set(row[o9Constants.SYSTEM_ENSEMBLE_ALGORITHM_LIST].split(","))
            system_assigned_algo = set(row[o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST].split(","))
            planner_assigned_algo = set(row[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST].split(","))
            if system_ensemble_algo.issubset(system_assigned_algo):
                return row[o9Constants.ASSIGNED_RULE]
            elif system_ensemble_algo.issubset(system_assigned_algo.union(planner_assigned_algo)):
                return "Planner Override"
        elif (
            row[o9Constants.STAT_ALGORITHM] in row[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST]
            or row[o9Constants.STAT_ALGORITHM] in row[o9Constants.PLANNER_BESTFIT_ALGORITHM]
        ):
            return "Planner Override"
        return "Custom"
    except Exception:
        logger.info("Planner Assigned Lists empty, returning Assigned rule")
        return row[o9Constants.ASSIGNED_RULE]


def add_dim_suffix(input: str, dim: str) -> str:
    if input == "All":
        return " ".join([input, dim])
    else:
        return input


def get_first_day_in_time_bucket(
    time_bucket_value: str,
    relevant_time_name: str,
    time_dimension: pd.DataFrame,
):
    # filter time dim data for the bucket value specified
    filter_clause = time_dimension[relevant_time_name] == time_bucket_value

    relevant_data = time_dimension[filter_clause]

    # sort by day key
    relevant_data.sort_values(o9Constants.DAY_KEY, inplace=True)

    return relevant_data[o9Constants.DAY].iloc[0]


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


def create_algo_list(AlgoDF: pd.DataFrame, algo_list_col: str, use_all_algos: bool = False) -> list:
    if use_all_algos:
        return get_algo_ranking()

    if len(AlgoDF) == 0:
        logger.warning("No Algorithms were assigned...")
        logger.warning("Returning empty list...")
        return []

    algos = AlgoDF[algo_list_col].iloc[0]

    # Split the string on comma
    algo_list = algos.split(",")
    # Use a list comprehension to remove any leading or trailing whitespaces
    algo_list = [item.strip() for item in algo_list]

    algo_list_cleaned = list(set(algo_list))
    # algo_list_cleaned = [item for item in algo_list if item not in algo_list_cleaned]

    return algo_list_cleaned


def list_of_cols_from_pyspark_schema(schema: StructType) -> list:
    return [field.name for field in schema]


def get_abs_error(
    source_df=pd.DataFrame(),
    Actuals=pd.DataFrame(),
    TimeDimension=pd.DataFrame(),
    CurrentTimePeriod=pd.DataFrame(),
    source_measure="Source Measure",
    actual_measure="Actual",
    output_measure="Output Measure",
    cycle_period=None,
    time_grain="Time.[Week]",
    time_key="Time.[WeekKey]",
    lags="0,1",
    six_cycles_flag=False,
    merge_grains=[],
    output_cols=[],
    df_keys={},
):
    """
    returns abs error on the time grain of the source dataframe
    source_df : Dataframe = Contains the forecast values at the desired time level, for which the abs error values are to be calculated
    cycle_period : str = if output needs to be filtered for any specific cycle periods
    lags : str = for filtering the lags needed by the user
    six_cycles_flag : bool = limits the output to last 6 cycles if set to True, in case cycle_period input is None


    """
    try:
        assert len(source_df) > 0, "Source Dataframe cannot be empty ..."
        assert len(Actuals) > 0, f"Actuals empty, returning empty outputs for slice {df_keys}"
        assert len(TimeDimension) > 0, "Time mapping df cannot be empty ..."
        assert (
            len(CurrentTimePeriod.drop_duplicates()) == 1
        ), "Current Time Period empty or Specific Period not given ..."
        assert (
            source_measure in source_df.columns
        ), "input measure incorrect, missing, or not part of the input dataframe ..."
        assert (
            actual_measure in Actuals.columns
        ), "Actual measure incorrect, missing, or not part of the Actual dataframe ..."
        assert (
            time_grain in source_df.columns
        ), "input time grain incorrect, missing, or not part of the input dataframe ..."

        relevant_time_grains = [
            o9Constants.PARTIAL_WEEK,
            o9Constants.PARTIAL_WEEK_KEY,
        ]

        if time_grain != o9Constants.PARTIAL_WEEK:
            relevant_time_grains = [
                o9Constants.PARTIAL_WEEK,
                o9Constants.PARTIAL_WEEK_KEY,
                time_grain,
                time_key,
            ]
        Actuals = Actuals.merge(
            TimeDimension[relevant_time_grains].drop_duplicates(),
            on=o9Constants.PARTIAL_WEEK,
            how="inner",
        )
        dimensions = merge_grains.copy()
        merge_grains += [time_grain]
        group_grains = merge_grains.copy()
        latest_time_period = CurrentTimePeriod[time_grain][0]
        if (
            o9Constants.LAG in source_df.columns
            and o9Constants.PLANNING_CYCLE_DATE in source_df.columns
        ):
            group_grains += [
                o9Constants.LAG,
                o9Constants.PLANNING_CYCLE_DATE,
            ]
            relevant_time_mapping = TimeDimension[[time_grain, time_key]].drop_duplicates()
            time_attribute_dict = {time_grain: time_key}
            if cycle_period:
                cycle_period_list = list(map(int, cycle_period.split(",")))

                last_n_time_periods = get_n_time_periods(
                    latest_time_period,
                    -max(cycle_period_list),
                    relevant_time_mapping,
                    time_attribute_dict,
                    include_latest_value=False,
                )
                last_cycle = last_n_time_periods[-1]
                Actuals = Actuals[Actuals[time_grain].isin(last_n_time_periods)]
                source_df = source_df[source_df[time_grain].isin(last_n_time_periods)]

                # recalculating cycle periods based on the previous cycle
                cycle_period_list = [i - 1 for i in cycle_period_list]
                planning_cycles = source_df[
                    (source_df[time_grain] == last_cycle)
                    & (source_df[o9Constants.LAG].astype(int).isin(cycle_period_list))
                ][o9Constants.PLANNING_CYCLE_DATE].drop_duplicates()
                source_df = source_df.merge(
                    planning_cycles,
                    on=o9Constants.PLANNING_CYCLE_DATE,
                    how="inner",
                )
            else:
                lags = lags.replace(" ", "").split(",")
                source_df = source_df[source_df[o9Constants.LAG].isin(lags)]
            source_intersections = source_df[dimensions].drop_duplicates()
            lag_intersections = source_df[
                [time_grain, o9Constants.LAG, o9Constants.PLANNING_CYCLE_DATE]
            ].drop_duplicates()
            all_intersections = create_cartesian_product(source_intersections, lag_intersections)
            if not all_intersections.empty:
                source_df = source_df.merge(all_intersections, on=group_grains, how="right")

        Actuals[actual_measure] = Actuals.groupby(merge_grains)[actual_measure].transform("sum")
        Actuals = Actuals.reset_index()

        source_df = source_df.merge(
            Actuals,
            on=merge_grains,
            how="outer",
        )
        source_df_nulls = source_df.loc[
            (source_df[source_measure].isna()) & (source_df[actual_measure].isna())
        ]
        source_df = source_df[~source_df.index.isin(source_df_nulls.index)]

        source_df[source_measure].fillna(0, inplace=True)
        source_df[actual_measure].fillna(0, inplace=True)

        source_df = source_df[
            group_grains
            + [
                actual_measure,
                source_measure,
            ]
        ].drop_duplicates()

        source_df = source_df.reset_index()

        source_df.loc[source_df[actual_measure] < 0, actual_measure] = 0

        source_df[output_measure] = abs(source_df[source_measure] - source_df[actual_measure])
        source_df = concat_to_dataframe([source_df, source_df_nulls])
        source_df = source_df[output_cols]
    except Exception as e:
        logger.warning(f"{e}")
        source_df = pd.DataFrame(columns=output_cols)
    return source_df


def get_grains_and_measures_from_dataset(list_of_cols: list):
    grain_list = [get_clean_string(col) for col in list_of_cols if is_dimension(col)]

    measure_list = [col for col in list_of_cols if not is_dimension(col)]
    return grain_list, measure_list


def check_duplicates(sdf: pyspark.sql.dataframe.DataFrame, cols: list = []):
    """Function to check if the spark df contains duplicates

    Args:
        df (pyspark.sql.dataframe.DataFrame): input spark df

    Raises:
        ValueError: Raises error if the data has duplicates
    """
    if len(cols) > 0:
        sdf = sdf.select(*cols)
    if sdf.count() > sdf.dropDuplicates().count():
        raise ValueError("Data has duplicates")


def insert_forecast_iteration_polars(df: pl.DataFrame, the_iteration: str) -> pl.DataFrame:
    if df.is_empty() and o9Constants.FORECAST_ITERATION not in df.columns:
        new_column_list = [o9Constants.FORECAST_ITERATION] + list(df.columns)
        df = pl.DataFrame({col: [] for col in new_column_list})
    else:
        if o9Constants.FORECAST_ITERATION not in df.columns:
            df = df.with_columns(pl.lit(the_iteration).alias(o9Constants.FORECAST_ITERATION))
            # Reorder so the new column is first
            df = df.select(
                [o9Constants.FORECAST_ITERATION]
                + [col for col in df.columns if col != o9Constants.FORECAST_ITERATION]
            )
    return df


def filter_for_iteration_polars(iteration: str):
    def decorator(func):
        def wrapper(**kwargs):
            logger.info(f"Filtering input dataframes for iteration {iteration}")

            # Filter dataframes if they are provided
            filtered_args = dict()
            for the_arg, the_value in kwargs.items():

                # exception : DP016TransitionLevelStat - StatFcstL1ForFIPLIteration - this will have data for FI-PL iteration but scope for the plugin would be some other iteration
                # do not filter this input dataframe for iteration
                if the_arg == "StatFcstL1ForFIPLIteration":
                    filtered_args[the_arg] = the_value
                    continue

                if isinstance(the_value, pl.DataFrame):
                    if o9Constants.FORECAST_ITERATION in the_value.columns:
                        # filter data for the iteration
                        filtered_df = the_value.filter(
                            pl.col(o9Constants.FORECAST_ITERATION) == iteration
                        ).clone()

                        # remove iteration column to avoid join related issues inside the plugin
                        filtered_df = filtered_df.drop(o9Constants.FORECAST_ITERATION)

                        # add it back with the same key
                        filtered_args[the_arg] = filtered_df

                    elif o9Constants.FORECAST_ITERATION_SELECTION in the_value.columns:
                        # remove nulls if present
                        filtered_df = the_value.filter(
                            pl.col(o9Constants.FORECAST_ITERATION_SELECTION).is_not_null()
                        )

                        # filter data for the iteration
                        filtered_df = filtered_df.filter(
                            pl.col(o9Constants.FORECAST_ITERATION_SELECTION).str.contains(iteration)
                        )
                        filtered_args[the_arg] = filtered_df

                    else:
                        filtered_args[the_arg] = the_value
                else:
                    filtered_args[the_arg] = the_value

            # call the function with the updated arguments
            # todo processIteration
            result = func(**filtered_args)
            # this should return the elements as polars from processIteration

            logger.info(f"Adding {o9Constants.FORECAST_ITERATION} : {iteration} to output ...")
            if isinstance(result, pl.DataFrame):

                # check and insert forecast iteration column
                result = insert_forecast_iteration_polars(df=result, the_iteration=iteration)
            elif isinstance(result, tuple):
                # check and insert forecast iteration column for every dataframe
                result = [
                    insert_forecast_iteration_polars(df=pl.DataFrame(x), the_iteration=iteration)
                    for x in result
                ]
            #  change the function that is consuming this results
            return result

        return wrapper

    return decorator
