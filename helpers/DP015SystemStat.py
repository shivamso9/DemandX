import logging

import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import get_seasonal_periods
from o9Reference.common_utils.dataframe_utils import (
    add_columns_to_df,
    concat_to_dataframe,
)
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data
from statsforecast import StatsForecast
from statsforecast.models import NaNModel

from helpers.models import train_models_for_one_intersection
from helpers.o9Constants import o9Constants
from helpers.utils import get_rule
from helpers.utils_forecaster import build_model_map
from helpers.utils_polars import (
    create_algo_list,
    extract_algo_params_batch,
    filter_for_iteration_polars,
    get_default_algo_params_polars,
    get_last_time_period,
    get_n_time_periods,
)

logger = logging.getLogger("o9_logger")


# Current Stat Implementation of StatsForecast
col_mapping = {
    "Algorithm Parameters": str,
    "Validation Method": str,
    "Run Time": float,
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
    "Stat Fcst SES 80% LB": float,
    "Stat Fcst DES 80% LB": float,
    "Stat Fcst TES 80% LB": float,
    "Stat Fcst ETS 80% LB": float,
    "Stat Fcst Auto ARIMA 80% LB": float,
    "Stat Fcst sARIMA 80% LB": float,
    "Stat Fcst Prophet 80% LB": float,
    "Stat Fcst STLF 80% LB": float,
    "Stat Fcst Theta 80% LB": float,
    "Stat Fcst Croston 80% LB": float,
    "Stat Fcst TBATS 80% LB": float,
    "Stat Fcst AR-NNET 80% LB": float,
    "Stat Fcst Simple Snaive 80% LB": float,
    "Stat Fcst Weighted Snaive 80% LB": float,
    "Stat Fcst Growth Snaive 80% LB": float,
    "Stat Fcst Naive Random Walk 80% LB": float,
    "Stat Fcst Seasonal Naive YoY 80% LB": float,
    "Stat Fcst Moving Average 80% LB": float,
    "Stat Fcst Simple AOA 80% LB": float,
    "Stat Fcst Growth AOA 80% LB": float,
    "Stat Fcst Weighted AOA 80% LB": float,
    "Stat Fcst SES 80% UB": float,
    "Stat Fcst DES 80% UB": float,
    "Stat Fcst TES 80% UB": float,
    "Stat Fcst ETS 80% UB": float,
    "Stat Fcst Auto ARIMA 80% UB": float,
    "Stat Fcst sARIMA 80% UB": float,
    "Stat Fcst Prophet 80% UB": float,
    "Stat Fcst STLF 80% UB": float,
    "Stat Fcst Theta 80% UB": float,
    "Stat Fcst Croston 80% UB": float,
    "Stat Fcst TBATS 80% UB": float,
    "Stat Fcst AR-NNET 80% UB": float,
    "Stat Fcst Simple Snaive 80% UB": float,
    "Stat Fcst Weighted Snaive 80% UB": float,
    "Stat Fcst Growth Snaive 80% UB": float,
    "Stat Fcst Naive Random Walk 80% UB": float,
    "Stat Fcst Seasonal Naive YoY 80% UB": float,
    "Stat Fcst Moving Average 80% UB": float,
    "Stat Fcst Simple AOA 80% UB": float,
    "Stat Fcst Growth AOA 80% UB": float,
    "Stat Fcst Weighted AOA 80% UB": float,
    "Stat Fcst SCHM 80% UB": float,
    "Stat Fcst SCHM 80% LB": float,
}


def rename_col(col):
    if "-lo-" in col:
        base, p = col.rsplit("-lo-", 1)
        return f"{base} {p}% LB"
    if "-hi-" in col:
        base, p = col.rsplit("-hi-", 1)
        return f"{base} {p}% UB"
    return col


# Forecast Dataframe Processing to Handle 53rd Week
def process_forecast_dataframe(df, fiscal_week_col):
    try:
        df.reset_index(drop=True, inplace=True)
        has_week_53 = "W53" in df[fiscal_week_col].values

        if has_week_53:
            week_53_index = df[df[fiscal_week_col] == "W53"].index[0]

            if 0 < week_53_index < len(df) - 1:
                df_till_53rdweek = df.iloc[:week_53_index]
                df_from_53rdweek = df.iloc[week_53_index:]
                forecast_columns = [col for col in df.columns if "Fcst" in col]
                df_from_53rdweek[forecast_columns] = df_from_53rdweek[forecast_columns].shift()
                df_from_53rdweek[forecast_columns] = df_from_53rdweek[forecast_columns].fillna(
                    (
                        df_till_53rdweek[forecast_columns].iloc[-1]
                        + df_from_53rdweek[forecast_columns].iloc[1]
                    )
                    / 2
                )
                df = pd.concat([df_till_53rdweek, df_from_53rdweek], ignore_index=True)

            elif week_53_index == len(df) - 1:
                forecast_columns = [col for col in df.columns if "Fcst" in col]
                df.loc[week_53_index, forecast_columns] = df.iloc[week_53_index - 1][
                    forecast_columns
                ]

        df = df.drop(fiscal_week_col, axis=1)
        # Removing the actual row of current time from the Forecast Dataframe
        df = df.iloc[1:]

    except Exception as e:
        logger.exception(e)

    return df


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    history_measure,
    AlgoList,
    Actual,
    TimeDimension,
    ForecastParameters,
    AlgoParameters,
    CurrentTimePeriod,
    non_disc_intersections,
    ForecastGenTimeBucket,
    StatBucketWeight,
    MasterAlgoList,
    DefaultAlgoParameters,
    SeasonalIndices=None,
    UseHolidays="False",
    IncludeDiscIntersections="True",
    multiprocessing_num_cores=4,
    AlgoListMeasure="Assigned Algorithm List",
    df_keys={},
    history_period_col="History Period",
    SellOutOffset=None,
    model_params=False,
):
    try:
        AllForecastList = list()
        ForecastModelList = list()
        # Prepare forecast level (grain columns)
        forecast_level = [x.strip() for x in Grains.split(",") if x.strip() not in ["", "NA"]]
        # Converting to polars dataframes
        Actual = pl.from_pandas(Actual)
        AlgoList = pl.from_pandas(AlgoList)
        TimeDimension = pl.from_pandas(TimeDimension)
        ForecastParameters = pl.from_pandas(ForecastParameters)
        AlgoParameters = pl.from_pandas(AlgoParameters)
        CurrentTimePeriod = pl.from_pandas(CurrentTimePeriod)
        non_disc_intersections = pl.from_pandas(non_disc_intersections)
        ForecastGenTimeBucket = pl.from_pandas(ForecastGenTimeBucket)
        StatBucketWeight = pl.from_pandas(StatBucketWeight)
        MasterAlgoList = pl.from_pandas(MasterAlgoList)
        DefaultAlgoParameters = pl.from_pandas(DefaultAlgoParameters)
        SeasonalIndices = pl.from_pandas(SeasonalIndices)
        SellOutOffset = pl.from_pandas(SellOutOffset)

        # gc.collect()  # Free memory from old pandas DataFrames

        for the_iteration in (
            ForecastGenTimeBucket.get_column(o9Constants.FORECAST_ITERATION).unique().to_list()
        ):
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration_polars(iteration=the_iteration)(processIteration)

            # Intersection Master Data Logic
            intersection_master = (
                Actual.select(forecast_level).unique().with_row_count("unique_id", offset=1)
            )

            Actual = Actual.join(intersection_master, on=forecast_level, how="left").drop(
                forecast_level
            )
            AlgoList = AlgoList.join(intersection_master, on=forecast_level, how="left").drop(
                forecast_level
            )
            AlgoParameters = AlgoParameters.join(
                intersection_master, on=forecast_level, how="left"
            ).drop(forecast_level)
            non_disc_intersections = non_disc_intersections.join(
                intersection_master, on=forecast_level, how="left"
            ).drop(forecast_level)
            SeasonalIndices = SeasonalIndices.join(
                intersection_master, on=forecast_level, how="left"
            ).drop(forecast_level)

            the_all_forecast, the_forecast_model = decorated_func(
                Grains=Grains,
                history_measure=history_measure,
                AlgoList=AlgoList,
                Actual=Actual,
                TimeDimension=TimeDimension,
                ForecastParameters=ForecastParameters,
                AlgoParameters=AlgoParameters,
                CurrentTimePeriod=CurrentTimePeriod,
                non_disc_intersections=non_disc_intersections,
                ForecastGenTimeBucket=ForecastGenTimeBucket,
                StatBucketWeight=StatBucketWeight,
                MasterAlgoList=MasterAlgoList,
                DefaultAlgoParameters=DefaultAlgoParameters,
                SeasonalIndices=SeasonalIndices,
                UseHolidays=UseHolidays,
                IncludeDiscIntersections=IncludeDiscIntersections,
                multiprocessing_num_cores=multiprocessing_num_cores,
                AlgoListMeasure=AlgoListMeasure,
                SellOutOffset=SellOutOffset,
                history_period_col=history_period_col,
                the_iteration=the_iteration,
                df_keys=df_keys,
                model_params=model_params,
            )

            AllForecastList.append(the_all_forecast)
            ForecastModelList.append(the_forecast_model)

        AllForecast = concat_to_dataframe(AllForecastList)
        ForecastModel = concat_to_dataframe(ForecastModelList)

        intersection_master = intersection_master.to_pandas()

        AllForecast = AllForecast.merge(intersection_master, on="unique_id", how="left").drop(
            "unique_id", axis=1
        )

        ForecastModel = ForecastModel.merge(intersection_master, on="unique_id", how="left").drop(
            "unique_id", axis=1
        )

        # Refactoring outputs
        version_col = "Version.[Version Name]"
        partial_week_col = "Time.[Partial Week]"
        assigned_algo_list_col = "Assigned Algorithm List"
        ALGO_MASTER_LIST = create_algo_list(
            AlgoDF=MasterAlgoList,
            algo_list_col=assigned_algo_list_col,
            use_all_algos=True,
        )

        ALGO_MASTER_LIST = [
            algo for algo in ALGO_MASTER_LIST if algo not in ["No Forecast", "Ensemble"]
        ]

        # Construct forecast column names
        ALL_STAT_FORECAST_COLUMNS = [f"Stat Fcst {algo}" for algo in ALGO_MASTER_LIST]
        ALL_STAT_FORECAST_LB_COLUMNS = [f"Stat Fcst {algo} 80% LB" for algo in ALGO_MASTER_LIST]
        ALL_STAT_FORECAST_UB_COLUMNS = [f"Stat Fcst {algo} 80% UB" for algo in ALGO_MASTER_LIST]

        ALL_FORECAST_COLUMNS = (
            ALL_STAT_FORECAST_COLUMNS + ALL_STAT_FORECAST_LB_COLUMNS + ALL_STAT_FORECAST_UB_COLUMNS
        )

        logger.info("Extracting forecast level ...")

        # Final column lists
        AllForecast_cols = (
            [o9Constants.FORECAST_ITERATION]
            + [version_col]
            + forecast_level
            + [partial_week_col]
            + ALL_FORECAST_COLUMNS
        )

        AllForecast = AllForecast[AllForecast_cols]

        stat_algo_col = "Stat Algorithm.[Stat Algorithm]"
        algorithm_parameters_col = "Algorithm Parameters"
        validation_method_col = "Validation Method"
        run_time_col = "Run Time"

        ForecastModel_cols = (
            [o9Constants.FORECAST_ITERATION]
            + [version_col]
            + forecast_level
            + [
                stat_algo_col,
                o9Constants.STAT_RULE,
                algorithm_parameters_col,
                validation_method_col,
                run_time_col,
            ]
        )

        ForecastModel = ForecastModel[ForecastModel_cols]

    except Exception as e:
        logger.exception(e)
        AllForecast, ForecastModel = None, None
    return AllForecast, ForecastModel


def processIteration(
    Grains,
    history_measure,
    AlgoList,
    Actual,
    TimeDimension,
    ForecastParameters,
    AlgoParameters,
    CurrentTimePeriod,
    non_disc_intersections,
    ForecastGenTimeBucket,
    StatBucketWeight,
    MasterAlgoList,
    DefaultAlgoParameters,
    the_iteration,
    SeasonalIndices=None,
    UseHolidays="False",
    IncludeDiscIntersections="True",
    multiprocessing_num_cores=4,
    AlgoListMeasure="Assigned Algorithm List",
    history_period_col="History Period",
    SellOutOffset=pd.DataFrame(),
    df_keys={},
    model_params=False,
):
    plugin_name = "DP015SystemStat"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    assigned_algo_list_col = "Assigned Algorithm List"
    default_stat_param_value_col = "Default Stat Parameter Value"
    history_period_col = "History Period"
    forecast_period_col = "Forecast Period"
    validation_period_col = "Validation Period"
    history_time_buckets_col = "History Time Buckets"
    stat_parameter_col = "Stat Parameter.[Stat Parameter]"
    stat_algo_col = "Stat Algorithm.[Stat Algorithm]"
    system_stat_param_value_col = "System Stat Parameter Value"
    best_fit_method_col = "Bestfit Method"
    confidence_interval_alpha = 0.20
    holiday_type_col = "Holiday Type"

    count_col = "Num of Datapoints"
    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    planning_month_col = "Time.[Planning Month]"
    quarter_col = "Time.[Quarter]"
    planning_quarter_col = "Time.[Planning Quarter]"

    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"

    partial_week_col = "Time.[Partial Week]"
    fiscal_week_col = "Time.[Week Name]"
    version_col = "Version.[Version Name]"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    stat_bucket_weight_col = "Stat Bucket Weight"
    prod_cust_segment_col = "Product Customer L1 Segment"
    algorithm_parameters_col = "Algorithm Parameters"
    validation_method_col = "Validation Method"
    run_time_col = "Run Time"
    stat_fcst_l1_flag = "Stat Fcst L1 Flag"
    trend_l1_col = "Trend L1"
    seasonality_l1_col = "Seasonality L1"
    sell_out_offset_col = "Offset Period"

    validation_seasonal_index_col = "SCHM Validation Seasonal Index"
    forward_seasonal_index_col = "SCHM Seasonal Index"
    forecast_level = "unique_id"

    UseHolidays = eval(UseHolidays)
    # Add names of new algorithms here
    # Extract only the algos that were assigned to intersections
    ALGO_MASTER_LIST = create_algo_list(
        AlgoDF=MasterAlgoList,
        algo_list_col=assigned_algo_list_col,
        use_all_algos=True,
    )

    # Remove "No Forecast" and "Ensemble" if present
    ALGO_MASTER_LIST = [
        algo for algo in ALGO_MASTER_LIST if algo not in ["No Forecast", "Ensemble"]
    ]

    logger.info(f"Master Algo List: {ALGO_MASTER_LIST}")

    # Construct forecast column names
    ALL_STAT_FORECAST_COLUMNS = [f"Stat Fcst {algo}" for algo in ALGO_MASTER_LIST]
    ALL_STAT_FORECAST_LB_COLUMNS = [f"Stat Fcst {algo} 80% LB" for algo in ALGO_MASTER_LIST]
    ALL_STAT_FORECAST_UB_COLUMNS = [f"Stat Fcst {algo} 80% UB" for algo in ALGO_MASTER_LIST]

    ALL_FORECAST_COLUMNS = (
        ALL_STAT_FORECAST_COLUMNS + ALL_STAT_FORECAST_LB_COLUMNS + ALL_STAT_FORECAST_UB_COLUMNS
    )

    logger.info("Extracting forecast level ...")

    # Final column lists
    AllForecast_cols = [version_col] + [forecast_level] + [partial_week_col] + ALL_FORECAST_COLUMNS

    # Empty Polars DataFrame with specified columns
    AllForecast = pl.DataFrame(schema={col: pl.Utf8 for col in AllForecast_cols})

    ForecastModel_cols = (
        [version_col]
        + [forecast_level]
        + [
            stat_algo_col,
            o9Constants.STAT_RULE,
            algorithm_parameters_col,
            validation_method_col,
            run_time_col,
        ]
    )
    ForecastModel = pl.DataFrame(schema={col: pl.Utf8 for col in ForecastModel_cols})

    try:
        if len(ALGO_MASTER_LIST) == 0:
            return pd.DataFrame(columns=AllForecast_cols), pd.DataFrame(columns=ForecastModel_cols)

        # Convert IncludeDiscIntersections from string to boolean safely
        if IncludeDiscIntersections not in ["True", "False"]:
            raise ValueError(
                f"'{IncludeDiscIntersections}' is invalid. Allowed values are 'True' or 'False'"
            )
        IncludeDiscIntersections = IncludeDiscIntersections == "True"

        logger.info(f"history_measure : {history_measure}")
        logger.info(f"AlgoListMeasure : {AlgoListMeasure}")

        # Filter out nulls and "No Forecast" entries
        AlgoList = AlgoList.filter(pl.col(AlgoListMeasure).is_not_null()).filter(
            ~pl.col(AlgoListMeasure).str.contains("No Forecast")
        )

        # Collect trend & seasonality into a separate DataFrame
        trend_seasonality_df = AlgoList.select(
            [forecast_level] + [trend_l1_col, seasonality_l1_col]
        ).unique()

        # Collect assigned rule, pbf, paal into another DataFrame
        assigned_rule = AlgoList.select(
            [forecast_level]
            + [
                o9Constants.ASSIGNED_RULE,
                o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST,
                o9Constants.PLANNER_BESTFIT_ALGORITHM,
                o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST,
            ]
        )

        logger.info("Filtering relevant columns from input dataframes ...")

        # Sellout offset handling

        if SellOutOffset.is_empty():
            logger.warning(
                f"Empty SellOut offset input for the forecast iteration {the_iteration}, assuming offset as 0 ..."
            )

            SellOutOffset = pl.DataFrame(
                {
                    o9Constants.VERSION_NAME: [
                        ForecastGenTimeBucket.select(o9Constants.VERSION_NAME).item()
                    ],
                    sell_out_offset_col: [0],
                }
            )

        if AlgoListMeasure == "Bestfit Algorithm":
            logger.info(f"AlgoList rows: {AlgoList.height}, columns: {AlgoList.width}")
            logger.info(
                f"Filtering rows where {stat_fcst_l1_flag} is 0, i.e forecast is not present ..."
            )

            AlgoList = AlgoList.filter(pl.col(stat_fcst_l1_flag) == 0)

            logger.info(f"AlgoList rows: {AlgoList.height}, columns: {AlgoList.width}")

        AlgoList = (
            AlgoList.lazy()
            .select([version_col] + [forecast_level] + [AlgoListMeasure])
            .filter(pl.col(AlgoListMeasure).is_not_null())
            .rename({AlgoListMeasure: assigned_algo_list_col})
        )

        if not IncludeDiscIntersections and not non_disc_intersections.is_empty():
            non_disc_intersections_unique = (
                non_disc_intersections.lazy()
                .filter(pl.col(prod_cust_segment_col).is_not_null())
                .select([forecast_level])
                .unique()
            )
            AlgoList = AlgoList.join(non_disc_intersections_unique, on=forecast_level, how="inner")

        AlgoList = AlgoList.collect()
        if AlgoList.is_empty():
            logger.warning(
                "No records found in AlgoList variable for slice : {} ...".format(df_keys)
            )
            logger.warning(
                "Kindly run 'Assign Rule and Algo' action button from Forecast Setup screen and trigger this plugin again ..."
            )
            return pd.DataFrame(columns=AllForecast_cols), pd.DataFrame(columns=ForecastModel_cols)

        # ForecastParameters filtering
        ForecastParameters = ForecastParameters.select(
            [
                version_col,
                history_period_col,
                forecast_period_col,
                validation_period_col,
                history_time_buckets_col,
                best_fit_method_col,
            ]
        )

        logger.info(
            f"Filtered ForecastParameters → rows: {ForecastParameters.height}, columns: {ForecastParameters.width}"
        )

        # AlgoParameters filtering
        AlgoParameters = AlgoParameters.select(
            [
                version_col,
                stat_parameter_col,
                forecast_level,
                stat_algo_col,
                system_stat_param_value_col,
            ]
        )

        logger.info(
            f"Filtered AlgoParameters → rows: {AlgoParameters.height}, columns: {AlgoParameters.width}"
        )

        # SeasonalIndices filtering (if present)
        if SeasonalIndices is not None:
            SeasonalIndices = SeasonalIndices.select(
                [
                    version_col,
                    forecast_level,
                    partial_week_col,
                    validation_seasonal_index_col,
                    forward_seasonal_index_col,
                ]
            )
            logger.info(
                f"Filtered SeasonalIndices → rows: {SeasonalIndices.height}, columns: {SeasonalIndices.width}"
            )

        # Combine checks:
        if (
            Actual is None
            or Actual.is_empty()
            or Actual.select(pl.col(history_measure).sum()).item() == 0
        ):
            logger.warning(f"Actuals is None/Empty or sum is zero for slice : {df_keys}...")
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=AllForecast_cols), pd.DataFrame(columns=ForecastModel_cols)

        # Extract values from ForecastParameters
        (history_periods, forecast_periods, validation_periods, frequency, validation_method) = (
            ForecastParameters.select(
                [
                    pl.col(history_period_col),
                    pl.col(forecast_period_col),
                    pl.col(validation_period_col),
                    pl.col(history_time_buckets_col),
                    pl.col(best_fit_method_col),
                ]
            ).row(0)
        )

        input_version = Actual.select(pl.col(version_col))[0].item()

        # Get forecast time bucket (Polars idiom)
        fcst_gen_time_bucket = ForecastGenTimeBucket.get_column(fcst_gen_time_bucket_col).unique()[
            0
        ]
        logger.debug(f"fcst_gen_time_bucket : {fcst_gen_time_bucket}")

        if TimeDimension.is_empty():
            logger.warning("TimeDimension is empty, returning empty dataframe")
            return pd.DataFrame(columns=AllForecast_cols), pd.DataFrame(columns=ForecastModel_cols)

        # Map columns
        bucket_map = {
            "Week": ("Weekly", [partial_week_col, week_col, week_key_col], week_col, week_key_col),
            "Planning Month": (
                "Monthly",
                [partial_week_col, planning_month_col, planning_month_key_col],
                planning_month_col,
                planning_month_key_col,
            ),
            "Month": (
                "Monthly",
                [partial_week_col, month_col, month_key_col],
                month_col,
                month_key_col,
            ),
            "Planning Quarter": (
                "Quarterly",
                [partial_week_col, planning_quarter_col, planning_quarter_key_col],
                planning_quarter_col,
                planning_quarter_key_col,
            ),
            "Quarter": (
                "Quarterly",
                [partial_week_col, quarter_col, quarter_key_col],
                quarter_col,
                quarter_key_col,
            ),
        }

        if fcst_gen_time_bucket not in bucket_map:
            logger.warning(
                f"Unknown fcst_gen_time_bucket {fcst_gen_time_bucket}, returning empty df"
            )
            return pd.DataFrame(columns=AllForecast_cols), pd.DataFrame(columns=ForecastModel_cols)

        frequency, relevant_time_cols, relevant_time_name, relevant_time_key = bucket_map[
            fcst_gen_time_bucket
        ]
        logger.debug(f"frequency : {frequency}")
        logger.debug(f"relevant_time_cols : {relevant_time_cols}")

        # Use LazyFrame for deduplication and filtering
        base_time_mapping = TimeDimension.lazy().select(relevant_time_cols).unique().collect()
        relevant_time_mapping = (
            TimeDimension.lazy().select([relevant_time_name, relevant_time_key]).unique().collect()
        )
        fiscalweeks_to_remove = (
            TimeDimension.lazy()
            .filter(pl.col(fiscal_week_col) == "W53")
            .select([o9Constants.PARTIAL_WEEK])
            .unique()
            .collect()
            .get_column(o9Constants.PARTIAL_WEEK)
            .to_list()
        )

        logger.info("---------------------------------------")
        logger.info("CurrentTimePeriod head : ")
        logger.info(
            CurrentTimePeriod.head() if hasattr(CurrentTimePeriod, "head") else CurrentTimePeriod
        )

        # Get latest time
        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )

        # Adjust for offset
        offset_periods = int(SellOutOffset[sell_out_offset_col][0])
        if offset_periods > 0:
            offset_time_periods = get_n_time_periods(
                latest_time_name,
                -int(offset_periods),
                relevant_time_mapping,
                relevant_time_name,
                relevant_time_key,
                include_latest_value=False,
            )
            latest_time_name = offset_time_periods[0]

        logger.info(f"Getting last {history_periods} period dates ...")

        last_n_periods = get_n_time_periods(
            latest_time_name,
            -int(history_periods),
            relevant_time_mapping,
            relevant_time_name,
            relevant_time_key,
        )

        forecast_period_dates = get_n_time_periods(
            latest_time_name,
            int(forecast_periods + offset_periods),
            relevant_time_mapping,
            relevant_time_name,
            relevant_time_key,
            include_latest_value=False,
        )

        if not last_n_periods:
            logger.warning(
                f"No dates found after filtering for {history_periods} periods for slice {df_keys}"
            )
            logger.warning("Returning empty dataframe ...")
            return pd.DataFrame(columns=AllForecast_cols), pd.DataFrame(columns=ForecastModel_cols)

        # change this threshold to let's say 3 if we don't want to generate forecast for intersections with 3 datapoints or less
        min_datapoints_required = 1

        cumulative_sum_col = f"Cumulative_{history_measure}"

        relevant_actuals = (
            Actual.with_columns(
                pl.when(pl.col(history_measure) < 0)
                .then(0)
                .otherwise(pl.col(history_measure))
                .alias(history_measure)
            )
            .select([version_col] + [forecast_level] + [partial_week_col, history_measure])
            .filter(pl.col(history_measure).is_not_null())
            .lazy()
            .join(base_time_mapping.lazy(), on=partial_week_col, how="inner")
            .filter(~pl.col(o9Constants.PARTIAL_WEEK).is_in(fiscalweeks_to_remove))
            .group_by([forecast_level] + [relevant_time_name])
            .agg(pl.col(history_measure).sum().alias(history_measure))
            .filter(pl.col(relevant_time_name).is_in(last_n_periods))
            .join(AlgoList.drop([version_col]).lazy(), on=forecast_level, how="inner")
            .with_columns(pl.count().over(forecast_level).alias(count_col))
            .filter(pl.col(count_col) >= min_datapoints_required)
            .drop([count_col])
            .join(relevant_time_mapping.lazy(), on=relevant_time_name, how="inner")
            .sort([forecast_level] + [relevant_time_key])
            .with_columns(
                pl.col(history_measure)
                .fill_null(0)
                .cum_sum()
                .over(forecast_level)
                .alias(cumulative_sum_col)
            )
            .filter(pl.col(cumulative_sum_col) > 0)
            .sort([forecast_level] + [relevant_time_key])
            .collect()
        )

        # If relevant_actuals is empty, return empty dataframes
        if relevant_actuals.is_empty():
            logger.warning("relevant_actuals is empty, returning empty dataframes ...")
            return pd.DataFrame(columns=AllForecast_cols), pd.DataFrame(columns=ForecastModel_cols)

        logger.info(f"relevant_actuals shape : {relevant_actuals.shape}")
        logger.info(f"relevant actuals head  : {relevant_actuals.head(5)}")

        # Getting seasonal periods
        seasonal_periods = get_seasonal_periods(frequency)

        if not AlgoParameters.is_empty():
            model_params = True

        # Check if model_params is False, then run without parameters
        if model_params is False:

            param_dict = extract_algo_params_batch(
                df=DefaultAlgoParameters,
                version=input_version,
                granularity=frequency,
                seasonal_periods=seasonal_periods,
            )

            model_map = build_model_map(param_dict, seasonal_periods)

            relevant_actuals = (
                relevant_actuals.with_columns(
                    pl.col(relevant_time_key).cast(pl.Datetime("ns")).alias(relevant_time_key)
                )
                .sort(["unique_id", relevant_time_key])
                .drop([relevant_time_name])
            )

            freq_dict = {"Weekly": "1w", "Monthly": "1mo", "Quarterly": "1q", "Yearly": "1y"}

            logger.info("________________________________________________")
            logger.info("Starting NIXTLA Models Bulk Forecasting ...")
            logger.info("________________________________________________")

            # Find all unique AlgoLists (as strings)
            unique_algosets = (
                relevant_actuals.select("Assigned Algorithm List")
                .unique()
                .get_column("Assigned Algorithm List")
                .to_list()
            )

            forecast_dfs = []
            for algo_list_str in unique_algosets:
                # Vectorized filter: select all rows where Assigned Algorithm List matches
                df_group = relevant_actuals.filter(
                    pl.col("Assigned Algorithm List") == algo_list_str
                )
                ids_in_group = df_group["unique_id"].unique().to_list()

                logger.info(f"No of ids in the group {len(ids_in_group)}")

                # **Split the algo list string to a list**
                model_names = [name.strip() for name in algo_list_str.split(",")]

                models_for_group = [model_map[name] for name in model_names if name in model_map]
                logger.debug(f"Models for the group {models_for_group}")

                logger.info(f"Running models {model_names} for {len(ids_in_group)} series.")

                # Only run StatsForecast if there's at least one series in this group
                if len(ids_in_group) > 0:
                    sf = StatsForecast(
                        models=models_for_group,
                        freq=freq_dict.get(frequency, "1w"),
                        n_jobs=multiprocessing_num_cores,
                        verbose=False,
                        fallback_model=NaNModel(),
                    )
                    df_subset = df_group.select(
                        [forecast_level, relevant_time_key, history_measure]
                    )

                    forecast = sf.forecast(
                        df=df_subset,
                        h=int(forecast_periods),
                        # level=[80],
                        id_col=forecast_level,
                        time_col=relevant_time_key,
                        target_col=history_measure,
                    )

                    forecast_dfs.append(forecast)

                    logger.info(
                        f"Forecast for {len(ids_in_group)} series completed for models {model_names}."
                    )

            _AllForecast = pl.concat(forecast_dfs, how="diagonal")
            logger.info("________________________________________________")
            logger.info("Running NIXTLA Models Bulk Forecasting is Completed ...")
            logger.info("________________________________________________")

            if (relevant_time_name == o9Constants.PLANNING_MONTH) or (
                relevant_time_name == o9Constants.PLANNING_QUARTER
            ):
                fcst_steps = (
                    _AllForecast.group_by("unique_id")
                    .agg(
                        [
                            pl.all().exclude(relevant_time_key),
                            pl.arange(1, pl.count() + 1).alias("step"),
                        ]
                    )
                    .explode(pl.all().exclude("unique_id"))
                )
                forecast_period_dates_df = pl.DataFrame(
                    {relevant_time_name: forecast_period_dates}
                ).with_row_count("step", offset=1)
                AllForecast = fcst_steps.join(
                    forecast_period_dates_df, on="step", how="inner"
                ).to_pandas()
            else:
                AllForecast = (
                    _AllForecast.with_columns(
                        [
                            pl.col(col).clip(lower_bound=0).round(3).alias(col)
                            for col in _AllForecast.columns
                            if col not in [forecast_level, relevant_time_key]
                        ]
                    )
                    .rename({col: rename_col(col) for col in _AllForecast.columns})
                    .join(
                        TimeDimension.select([relevant_time_name, relevant_time_key])
                        .unique()
                        .with_columns(
                            pl.col(relevant_time_key)
                            .cast(pl.Datetime("ns"))
                            .alias(relevant_time_key)
                        ),
                        on=relevant_time_key,
                        how="left",
                    )
                    .drop(relevant_time_key)
                    .to_pandas()
                )

            ForecastModel = ForecastModel.to_pandas()

        else:
            logger.info("Running NIXTLA Models Forecasting with parameters ...")

            intersections_master = relevant_actuals.select(forecast_level).unique()

            if AlgoParameters.is_empty():
                logger.info(
                    "No AlgoParameters supplied, creating master list of algo params for all intersections ..."
                )

                AlgoParameters = get_default_algo_params_polars(
                    stat_algo_col=stat_algo_col,
                    stat_parameter_col=stat_parameter_col,
                    system_stat_param_value_col=system_stat_param_value_col,
                    frequency=frequency,
                    intersections_master=intersections_master,
                    DefaultAlgoParameters=DefaultAlgoParameters,
                )

            else:
                logger.info(
                    f"AlgoParameters supplied, shape: ({AlgoParameters.height}, {AlgoParameters.width})"
                )
                logger.info(
                    "Joining with default params to populate values for all intersections ..."
                )

                DefaultParameters = get_default_algo_params_polars(
                    stat_algo_col=stat_algo_col,
                    stat_parameter_col=stat_parameter_col,
                    system_stat_param_value_col=default_stat_param_value_col,
                    frequency=frequency,
                    intersections_master=intersections_master,
                    DefaultAlgoParameters=DefaultAlgoParameters,
                )

                DefaultParameters = DefaultParameters.with_columns(
                    pl.lit(input_version).alias(version_col)
                )

                # Compose join keys: algo params + intersection dimensions
                join_keys = [stat_algo_col, stat_parameter_col, forecast_level, version_col]

                # Merge and fill nulls with default, drop extra column
                AlgoParameters = (
                    AlgoParameters.join(DefaultParameters, on=join_keys, how="right")
                    .with_columns(
                        pl.when(pl.col(system_stat_param_value_col).is_null())
                        .then(pl.col(default_stat_param_value_col))
                        .otherwise(pl.col(system_stat_param_value_col))
                        .alias(system_stat_param_value_col)
                    )
                    .drop(default_stat_param_value_col)
                )

            relevant_actuals = (
                relevant_actuals.to_pandas()
                .merge(trend_seasonality_df.to_pandas(), on=forecast_level, how="left")
                .pipe(
                    lambda df: pd.concat(
                        [
                            df,
                            intersections_master.to_pandas().merge(
                                pd.DataFrame({relevant_time_name: forecast_period_dates}),
                                how="cross",
                            ),
                        ],
                        ignore_index=True,
                    )
                )
                .assign(
                    **{
                        holiday_type_col: "NA",
                        validation_seasonal_index_col: 1,
                        forward_seasonal_index_col: 1,
                    }
                )
            )

            AlgoParameters = AlgoParameters.to_pandas()

            all_results = Parallel(n_jobs=1, verbose=1)(
                delayed(train_models_for_one_intersection)(
                    df=df,
                    forecast_level=[forecast_level],
                    relevant_time_name=relevant_time_name,
                    history_measure=history_measure,
                    validation_periods=history_periods,
                    validation_method=validation_method,
                    seasonal_periods=seasonal_periods,
                    forecast_period_dates=forecast_period_dates,
                    confidence_interval_alpha=confidence_interval_alpha,
                    assigned_algo_list_col=assigned_algo_list_col,
                    AlgoParameters=AlgoParameters,
                    stat_algo_col=stat_algo_col,
                    stat_parameter_col=stat_parameter_col,
                    system_stat_param_value_col=system_stat_param_value_col,
                    holiday_type_col=holiday_type_col,
                    use_holidays="False",
                    validation_seasonal_index_col=validation_seasonal_index_col,
                    forward_seasonal_index_col=forward_seasonal_index_col,
                    trend_col=trend_l1_col,
                    seasonality_col=seasonality_l1_col,
                )
                for name, df in relevant_actuals.groupby(forecast_level)
            )

            logger.info("Collected results from parallel processing ...")

            # collect separate lists from the list of tuples returned by multiprocessing function
            all_forecasts = [x[1] for x in all_results]
            all_model_descriptions = [x[2] for x in all_results]
            # Intersections with Algo Parameters
            AllForecast = concat_to_dataframe(all_forecasts)
            ForecastModel = concat_to_dataframe(all_model_descriptions)

            if len(ForecastModel) == 0:
                ForecastModel = pd.DataFrame(columns=ForecastModel_cols)
            else:
                ForecastModel = (
                    ForecastModel.merge(assigned_rule.to_pandas(), on=forecast_level, how="inner")
                    .assign(**{o9Constants.STAT_RULE: lambda df: df.apply(get_rule, axis=1)})
                    .assign(**{version_col: input_version})
                    .loc[:, ForecastModel_cols]
                )

        # Disaggregate Forecast from Forecast Gen Time Bucket to Partial Week
        AllForecast = (
            AllForecast.assign(**{version_col: input_version})
            .pipe(add_columns_to_df, list_of_cols=ALL_FORECAST_COLUMNS)
            .pipe(
                lambda df: disaggregate_data(
                    source_df=df,
                    source_grain=relevant_time_name,
                    target_grain=partial_week_col,
                    profile_df=StatBucketWeight.join(
                        base_time_mapping, on=partial_week_col, how="inner"
                    )
                    .to_pandas()
                    .drop(version_col, axis=1),
                    profile_col=stat_bucket_weight_col,
                    cols_to_disaggregate=ALL_FORECAST_COLUMNS,
                )
            )
            .loc[:, AllForecast_cols]
        )

        logger.info("Successfully executed {} ...".format(plugin_name))
        logger.info("---------------------------------------------")
        logger.info("Processing completed for all intersections ...")
    except Exception as e:
        logger.error("Error in Data Pipeline")
        logger.exception(f"Exception {e} for slice : {df_keys}")
        AllForecast = pd.DataFrame(columns=AllForecast_cols)
        ForecastModel = pd.DataFrame(columns=ForecastModel_cols)

    return AllForecast, ForecastModel
