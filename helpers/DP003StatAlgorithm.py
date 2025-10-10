import logging
import math

import numpy as np
import pandas as pd
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
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.time_series import calculate_trend
from statsmodels.tsa.stattools import acf

from helpers.disaggregation import join_lowest_level
from helpers.DP015IdentifyBestFitModel import processIteration as identify_best_fit
from helpers.DP015PopulateBestFitForecast import (
    processIteration as populate_best_fit_forecast,
)
from helpers.model_params import get_default_algo_params
from helpers.models import train_models_for_one_intersection
from helpers.o9Constants import o9Constants
from helpers.utils import get_measure_name, get_seasonal_periods

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def get_measure_name_custom(key: str):
    return " ".join(["Stat Forecast", key])


def ACFDetector(df_sub, measure_name, freq, skip_lags, diff, alpha):
    ts_diff = df_sub[measure_name].values

    # Parameters
    lags = get_seasonal_periods(freq)  # weekly = 52

    lower_ci_threshold = -0.05
    upper_ci_threshold = 0.95

    for _ in range(diff):
        ts_diff = np.diff(ts_diff)

    ac, confint, qstat, qval = acf(ts_diff, nlags=lags, qstat=True, alpha=alpha)
    # get seasonality cycle length
    raw_seasonality = []
    for i, _int in enumerate(confint):
        if ((_int[0] >= lower_ci_threshold) or (_int[1] >= upper_ci_threshold)) and i > 1:
            raw_seasonality.append(i)

    seasonality = get_seasonality_length(raw_seasonality, skip_lags)
    seasonality_detected = True if len(seasonality) >= 1 else False
    seasonality = ",".join(map(str, seasonality)) if seasonality_detected else None
    return seasonality_detected, seasonality


def get_seasonality_length(d, skip_lags):
    out = []

    if len(d) > 1:
        if all(np.diff(d) == np.diff(d)[0]):
            out.append(max(d))
            return out
    while d:
        k = d.pop(0)
        d = [i for i in d if i % k != 0]
        out.append(k)

    # Reducing the options to avoid consecutive (upto 2) lags
    out.sort(reverse=True)

    cleaned_out = []
    for val in out:
        if len(cleaned_out) < 1:
            cleaned_out.append(val)
        else:
            if cleaned_out[-1] - val <= skip_lags:
                pass
            else:
                cleaned_out.append(val)
    cleaned_out.sort(reverse=True)
    cleaned_out = cleaned_out[:3]  # Top 3 periods only

    return cleaned_out


def calc_trend(
    df,
    forecast_level,
    trend_col,
    los_col,
    frequency,
    history_measure,
    trend_threshold,
):
    result = df[forecast_level].drop_duplicates()

    # create fallback dataframe for results
    result[trend_col] = "NO TREND"
    trend_factor_col = "TrendFactor"

    if df[los_col].unique()[0] >= get_seasonal_periods(frequency):
        # calculate trend
        result[trend_factor_col] = calculate_trend(df[history_measure].to_numpy())

        # Assign trend category
        conditions = [
            result[trend_factor_col] > trend_threshold,
            (result[trend_factor_col] <= trend_threshold)
            & (result[trend_factor_col] >= -trend_threshold),
            result[trend_factor_col] < -trend_threshold,
        ]
        choices = ["UPWARD", "NO TREND", "DOWNWARD"]

        logger.info("Assigning trend categories ...")
        result[trend_col] = np.select(conditions, choices, default=None)

    return result


def calc_seasonality(
    df,
    forecast_level,
    seasonality_col,
    los_col,
    frequency,
    history_measure,
    skip_lags,
    diff,
    alpha,
):
    result = df[forecast_level].drop_duplicates()

    # create fallback dataframe for results
    result[seasonality_col] = "Does not Exist"

    if (
        df[los_col].unique()[0] >= 2 * get_seasonal_periods(frequency)
        and df[history_measure].sum() > 0
    ):
        acf_seasonal_presence, acf_seasonalities = ACFDetector(
            df,
            history_measure,
            frequency,
            skip_lags,
            diff,
            alpha,
        )
        # Storage of results
        result[seasonality_col] = "Exists" if acf_seasonal_presence else "Does not Exist"

    return result


def add_trend_seasonality(
    df: pd.DataFrame,
    frequency: str,
    trend_col: str,
    seasonality_col: str,
    history_measure: str,
    forecast_level: list,
) -> pd.DataFrame:

    los_col = "Length of Series"

    # add los
    df[los_col] = df.groupby(forecast_level)[history_measure].transform("count")

    # Use default thresholds for trend and seasonality since this won't be coming from the tenant base configuration
    trend_threshold = 0.01

    # calculate trend
    trend_df = df.groupby(by=forecast_level, as_index=False).apply(
        lambda x: calc_trend(
            df=x,
            forecast_level=forecast_level,
            trend_col=trend_col,
            los_col=los_col,
            frequency=frequency,
            history_measure=history_measure,
            trend_threshold=trend_threshold,
        )
    )

    # calculate seasonality
    skip_lags = 2  # monthly
    diff = 1
    alpha = float(0.05)
    seasonality_df = df.groupby(forecast_level).apply(
        lambda x: calc_seasonality(
            df=x,
            forecast_level=forecast_level,
            seasonality_col=seasonality_col,
            los_col=los_col,
            frequency=frequency,
            history_measure=history_measure,
            skip_lags=skip_lags,
            diff=diff,
            alpha=alpha,
        )
    )
    seasonality_df.reset_index(drop=True, inplace=True)

    df = df.merge(trend_df, on=forecast_level, how="inner")

    df = df.merge(seasonality_df, on=forecast_level, how="inner")

    return df


col_mapping = {
    "Stat Forecast Description": str,
    "Stat Forecast Model 1": float,
    "Stat Forecast Model 2": float,
    "Stat Forecast Model 3": float,
    "Stat Forecast Best Fit": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    CurrentTimePeriod,
    Input_Attribute_PlanningItem,
    Input_Attribute_Location,
    Input_Cleansed_History,
    Input_Stat_Level,
    Input_Algorithm_Association,
    Input_Parameter_Value,
    TimeDimension,
    MasterAlgoList,
    Input_Attribute_Channel,
    Input_Attribute_Region,
    Input_Attribute_Account,
    Input_Attribute_PnL,
    Input_Attribute_DemandDomain,
    DefaultAlgoParameters,
    multiprocessing_num_cores=4,
    df_keys={},
):
    plugin_name = "DP003StatAlgorithm"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    stat_fcst_desc_col = "Stat Forecast Description"
    stat_fcst_model_1_col = "Stat Forecast Model 1"
    stat_fcst_model_2_col = "Stat Forecast Model 2"
    stat_fcst_model_3_col = "Stat Forecast Model 3"
    system_stat_fcst_l1_col = "System Stat Fcst L1"
    stat_fcst_best_fit_col = "Stat Forecast Best Fit"
    assigned_algo_list_col = "Assigned Algorithm List"
    L0ItemLevel = "Item.[Planning Item]"
    L0LocationLevel = "Location.[Planning Location]"
    L0ChannelLevel = "Channel.[Planning Channel]"
    L0RegionLevel = "Region.[Planning Region]"
    L0AccountLevel = "Account.[Planning Account]"
    L0PnLLevel = "PnL.[Planning PnL]"
    L0DemandDomainLevel = "Demand Domain.[Planning Demand Domain]"
    ActualMeasure = "Stat Cleansed History"
    version_col = "Version.[Version Name]"
    stat_model_col = "Stat Model.[Stat Model]"
    stat_algo_col = "Stat Algorithm.[Stat Algorithm]"
    stat_model_parameter_association_col = "Stat Model Parameter Association"
    stat_parameter_col = "Stat Parameter.[Stat Parameter]"
    system_stat_param_value_col = "System Stat Parameter Value"
    default_stat_param_value_col = "Default Stat Parameter Value"
    confidence_interval_alpha = 0.20
    validation_method = "Out Sample"
    error_metric = "RMSE"
    system_best_fit_algo_col = "System Bestfit Algorithm"
    planner_best_fit_algo_col = "Planner Bestfit Algorithm"
    future_periods_col = "Future Periods"
    group_sum_col = "Group Sum"
    split_percent_col = "Split Percentage"
    holiday_type_col = "Holiday Type"
    use_holidays = False
    stat_param_value_col = "Stat Parameter Value"
    history_period_col = "History Period"
    forecast_period_col = "Forecast Period"
    validation_period_col = "Validation Period"
    bestfit_method_col = "Bestfit Method"
    error_metric_col = "Error Metric"
    history_time_buckets_col = "History Time Buckets"
    week_key_col = "Time.[WeekKey]"
    month_key_col = "Time.[MonthKey]"
    planning_month_col = "Time.[Planning Month]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    partial_week_col = "Time.[Partial Week]"
    week_col = "Time.[Week]"
    month_col = "Time.[Month]"
    quarter_col = "Time.[Quarter]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_col = "Time.[Planning Quarter]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"
    fcst_gen_time_bucket_col = "Forecast Generation Time Bucket"
    day_col = "Time.[Day]"
    num_days_col = "Number of Days"
    total_num_days_col = "Total Number of Days"
    stat_bucket_weight_col = "Stat Bucket Weight"
    partial_week_key_col = "Time.[PartialWeekKey]"
    system_stat_fcst_l1_ub_col = system_stat_fcst_l1_col + " 80% UB"
    system_stat_fcst_l1_lb_col = system_stat_fcst_l1_col + " 80% LB"
    algorithm_parameters_col = "Algorithm Parameters"
    trend_col = "Trend"
    seasonality_col = "Seasonality"
    bestfit_selection_criteria_col = "Bestfit Selection Criteria"
    validation_seasonal_index_col = "SCHM Validation Seasonal Index"
    forward_seasonal_index_col = "SCHM Seasonal Index"

    OutputForecast_cols = [
        version_col,
        L0LocationLevel,
        L0ItemLevel,
        L0ChannelLevel,
        L0RegionLevel,
        L0AccountLevel,
        L0PnLLevel,
        L0DemandDomainLevel,
        partial_week_col,
        stat_fcst_best_fit_col,
        stat_fcst_model_1_col,
        stat_fcst_model_2_col,
        stat_fcst_model_3_col,
    ]
    OutputForecast = pd.DataFrame(columns=OutputForecast_cols)

    OutputDescription_cols = [
        version_col,
        L0LocationLevel,
        L0ItemLevel,
        L0ChannelLevel,
        L0RegionLevel,
        L0AccountLevel,
        L0PnLLevel,
        L0DemandDomainLevel,
        stat_model_col,
        stat_fcst_desc_col,
    ]
    OutputDescription = pd.DataFrame(columns=OutputDescription_cols)
    try:
        if Input_Algorithm_Association.empty:
            logger.warning(
                "Input_Algorithm_Association is empty, returning without further execution ..."
            )
            return OutputForecast, OutputDescription

        # Extract the interactive stat level
        ItemLevel = "".join(["Item.[", Input_Stat_Level["Stat Item Level"].iloc[0], "]"])
        LocationLevel = "".join(
            [
                "Location.[",
                Input_Stat_Level["Stat Location Level"].iloc[0],
                "]",
            ]
        )
        ChannelLevel = "".join(
            [
                "Channel.[",
                Input_Stat_Level["Stat Channel Level"].iloc[0],
                "]",
            ]
        )
        RegionLevel = "".join(
            [
                "Region.[",
                Input_Stat_Level["Stat Region Level"].iloc[0],
                "]",
            ]
        )
        AccountLevel = "".join(
            [
                "Account.[",
                Input_Stat_Level["Stat Account Level"].iloc[0],
                "]",
            ]
        )
        PnLLevel = "".join(
            [
                "PnL.[",
                Input_Stat_Level["Stat PnL Level"].iloc[0],
                "]",
            ]
        )
        DemandDomainLevel = "".join(
            [
                "Demand Domain.[",
                Input_Stat_Level["Stat Demand Domain Level"].iloc[0],
                "]",
            ]
        )
        TimeLevel = "".join(["Time.[", Input_Stat_Level["Stat Time Level"].iloc[0], "]"])

        logger.info(
            "Interactive Stat Level : {}".format(
                [
                    ItemLevel,
                    LocationLevel,
                    ChannelLevel,
                    AccountLevel,
                    RegionLevel,
                    PnLLevel,
                    DemandDomainLevel,
                    TimeLevel,
                ]
            )
        )

        # Default is week
        frequency = "Weekly"
        forecast_periods = 104
        validation_periods = 26
        history_periods = 52 * 3
        relevant_time_name = week_col
        relevant_time_key = week_key_col
        relevant_time_cols = [partial_week_col, week_col, week_key_col]
        fcst_gen_time_bucket = "Week"

        if TimeLevel == "Time.[Planning Month]":
            frequency = "Monthly"
            forecast_periods = 24
            validation_periods = 6
            history_periods = 12 * 3
            relevant_time_cols = [
                partial_week_col,
                planning_month_col,
                planning_month_key_col,
            ]
            relevant_time_name = planning_month_col
            relevant_time_key = planning_month_key_col
            fcst_gen_time_bucket = "Planning Month"
        elif TimeLevel == "Time.[Month]":
            frequency = "Monthly"
            forecast_periods = 24
            validation_periods = 6
            history_periods = 12 * 3
            relevant_time_cols = [partial_week_col, month_col, month_key_col]
            relevant_time_name = month_col
            relevant_time_key = month_key_col
            fcst_gen_time_bucket = "Month"
        elif TimeLevel == "Time.[Quarter]":
            frequency = "Quarterly"
            forecast_periods = 8
            validation_periods = 2
            history_periods = 4 * 3
            relevant_time_cols = [
                partial_week_col,
                quarter_col,
                quarter_key_col,
            ]
            relevant_time_name = quarter_col
            relevant_time_key = quarter_key_col
            fcst_gen_time_bucket = "Quarter"
        elif TimeLevel == "Time.[Planning Quarter]":
            frequency = "Quarterly"
            forecast_periods = 8
            validation_periods = 2
            history_periods = 4 * 3
            relevant_time_cols = [
                partial_week_col,
                planning_quarter_col,
                planning_quarter_key_col,
            ]
            relevant_time_name = planning_quarter_col
            relevant_time_key = planning_quarter_key_col
            fcst_gen_time_bucket = "Quarter"

        logger.info("TimeLevel : {}".format(TimeLevel))
        logger.info("frequency : {}".format(frequency))
        logger.info("forecast_periods : {}".format(forecast_periods))
        logger.info("validation_periods : {}".format(validation_periods))
        logger.info("history_periods : {}".format(history_periods))

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # Filter relevant columns from time mapping
        relevant_time_mapping = TimeDimension[
            [relevant_time_name, relevant_time_key]
        ].drop_duplicates()

        # Create flags for isLowest level
        IsLowest_ItemLevel = 1 if ItemLevel == L0ItemLevel else 0
        IsLowest_LocationLevel = 1 if LocationLevel == L0LocationLevel else 0
        IsLowest_ChannelLevel = 1 if ChannelLevel == L0ChannelLevel else 0
        IsLowest_RegionLevel = 1 if RegionLevel == L0RegionLevel else 0
        IsLowest_AccountLevel = 1 if AccountLevel == L0AccountLevel else 0
        IsLowest_PnLLevel = 1 if PnLLevel == L0PnLLevel else 0
        IsLowest_DemandDomainLevel = 1 if DemandDomainLevel == L0DemandDomainLevel else 0

        # Make sure the order of elements in all lists is same
        interactive_stat_level = [
            ItemLevel,
            LocationLevel,
            ChannelLevel,
            RegionLevel,
            AccountLevel,
            PnLLevel,
            DemandDomainLevel,
        ]
        lowest_level = [
            L0ItemLevel,
            L0LocationLevel,
            L0ChannelLevel,
            L0RegionLevel,
            L0AccountLevel,
            L0PnLLevel,
            L0DemandDomainLevel,
        ]
        dim_data = [
            Input_Attribute_PlanningItem,
            Input_Attribute_Location,
            Input_Attribute_Channel,
            Input_Attribute_Region,
            Input_Attribute_Account,
            Input_Attribute_PnL,
            Input_Attribute_DemandDomain,
        ]
        is_lowest = [
            IsLowest_ItemLevel,
            IsLowest_LocationLevel,
            IsLowest_ChannelLevel,
            IsLowest_RegionLevel,
            IsLowest_AccountLevel,
            IsLowest_PnLLevel,
            IsLowest_DemandDomainLevel,
        ]

        # Join cleansed history to get columns at interactive stat level
        history_with_higher_level_columns = join_lowest_level(
            df=Input_Cleansed_History,
            required_level=interactive_stat_level,
            lowest_level=lowest_level,
            dim_master_data=dim_data,
            is_lowest=is_lowest,
            join_on_lower_level=True,
        )

        # collect base time mapping
        base_time_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        # join with time mapping
        history_with_higher_level_columns = history_with_higher_level_columns.merge(
            base_time_mapping, on=partial_week_col, how="inner"
        )

        # groupby and sum at required level
        aggregated_history = (
            history_with_higher_level_columns.groupby(interactive_stat_level + [TimeLevel])
            .sum()[[ActualMeasure]]
            .reset_index()
        )

        # Create and maintain forecast level (excluding timelevel)
        forecast_level = [
            ItemLevel,
            LocationLevel,
            ChannelLevel,
            RegionLevel,
            AccountLevel,
            PnLLevel,
            DemandDomainLevel,
        ]

        # collect intersections master
        intersections_master = aggregated_history[forecast_level].drop_duplicates()

        if len(DefaultAlgoParameters) == 0:
            logger.warning("DefaultAlgoParameters is empty, returning without further execution")
            return OutputForecast, OutputDescription

        # Get list of algorithms to be run
        algorithms = list(Input_Algorithm_Association[stat_algo_col].unique())

        # Filter parameter values for those algorithms
        Input_Parameter_Value = Input_Parameter_Value[
            Input_Parameter_Value[stat_algo_col].isin(algorithms)
        ]
        Input_Parameter_Value.drop(stat_model_parameter_association_col, axis=1, inplace=True)

        input_version = Input_Algorithm_Association[version_col].iloc[0]

        # assign parameter values to all intersections
        Input_Parameter_Value_all_intersections = create_cartesian_product(
            df1=Input_Parameter_Value, df2=intersections_master
        )
        Input_Parameter_Value_all_intersections.rename(
            columns={stat_param_value_col: system_stat_param_value_col},
            inplace=True,
        )

        if len(Input_Parameter_Value) == 0:
            logger.info(
                "No AlgoParameters supplied, creating master list of algo params for all intersections ..."
            )
            AlgoParameters = get_default_algo_params(
                stat_algo_col,
                stat_parameter_col,
                system_stat_param_value_col,
                frequency,
                intersections_master,
                DefaultAlgoParameters,
            )

        else:
            logger.info(
                "AlgoParameters supplied, shape : {} ...".format(Input_Parameter_Value.shape)
            )
            logger.info("Joining with default params to populate values for all intersections ...")

            DefaultParameters = get_default_algo_params(
                stat_algo_col,
                stat_parameter_col,
                default_stat_param_value_col,
                frequency,
                intersections_master,
                DefaultAlgoParameters,
            )

            AlgoParameters = Input_Parameter_Value_all_intersections.merge(
                DefaultParameters, how="right"
            )

            AlgoParameters[system_stat_param_value_col] = np.where(
                AlgoParameters[system_stat_param_value_col].isna(),
                AlgoParameters[default_stat_param_value_col],
                AlgoParameters[system_stat_param_value_col],
            )

        # collect relevant columns
        algo_df = Input_Algorithm_Association[[stat_model_col, stat_algo_col]].drop_duplicates()

        # get last time period name
        last_time_period_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )
        logger.info("last_time_period_name : {}".format(last_time_period_name))

        seasonal_periods = get_seasonal_periods(frequency)

        # get forecast periods
        forecast_period_date_names = get_n_time_periods(
            latest_value=last_time_period_name,
            periods=forecast_periods,
            time_mapping=relevant_time_mapping,
            time_attribute=time_attribute_dict,
            include_latest_value=False,
        )

        # Create forecast parameters df as required by best fit script
        ForecastParameters = pd.DataFrame(
            {
                version_col: input_version,
                history_period_col: history_periods,
                forecast_period_col: forecast_periods,
                validation_period_col: validation_periods,
                bestfit_method_col: validation_method,
                error_metric_col: error_metric,
                history_time_buckets_col: frequency,
            },
            index=[0],
        )
        rename_mapping = {}
        rename_mapping_model = {}

        # loop on model
        for idx, row in algo_df.iterrows():
            model_num = row[stat_model_col]
            model = row[stat_algo_col]

            rename_mapping[get_measure_name(model)] = get_measure_name_custom(model_num)
            rename_mapping_model[model + " Model"] = model_num

        # get required history periods
        last_n_periods = get_n_time_periods(
            latest_value=last_time_period_name,
            periods=-history_periods,
            time_mapping=relevant_time_mapping,
            time_attribute=time_attribute_dict,
            include_latest_value=True,
        )

        # fill missing dates
        relevant_history_nas_filled = fill_missing_dates(
            actual=aggregated_history,
            forecast_level=forecast_level,
            time_mapping=relevant_time_mapping,
            history_measure=ActualMeasure,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            relevant_time_periods=last_n_periods,
            fill_nulls_with_zero=True,
        )

        # assign algo list to actuals dataframe itself
        relevant_history_nas_filled[assigned_algo_list_col] = ",".join(
            algo_df[stat_algo_col].unique()
        )

        # add holiday type colum
        relevant_history_nas_filled[holiday_type_col] = "NA"

        # sort data
        relevant_history_nas_filled.sort_values(forecast_level + [relevant_time_key], inplace=True)

        # assign trend/seasonality
        relevant_history_nas_filled = add_trend_seasonality(
            df=relevant_history_nas_filled,
            frequency=frequency,
            trend_col=trend_col,
            seasonality_col=seasonality_col,
            history_measure=ActualMeasure,
            forecast_level=forecast_level,
        )
        # For Backward compatibility adding seasonal indices and fill with 1
        if (
            validation_seasonal_index_col not in relevant_history_nas_filled.columns
            and forward_seasonal_index_col not in relevant_history_nas_filled.columns
        ):
            relevant_history_nas_filled[
                [validation_seasonal_index_col, forward_seasonal_index_col]
            ] = 1

        # run models for all intersections
        all_results = Parallel(n_jobs=1, verbose=1)(
            delayed(train_models_for_one_intersection)(
                df=df,
                forecast_level=forecast_level,
                relevant_time_name=TimeLevel,
                history_measure=ActualMeasure,
                validation_periods=validation_periods,
                validation_method=validation_method,
                seasonal_periods=seasonal_periods,
                forecast_period_dates=forecast_period_date_names,
                confidence_interval_alpha=confidence_interval_alpha,
                assigned_algo_list_col=assigned_algo_list_col,
                AlgoParameters=AlgoParameters,
                stat_algo_col=stat_algo_col,
                stat_parameter_col=stat_parameter_col,
                system_stat_param_value_col=system_stat_param_value_col,
                holiday_type_col=holiday_type_col,
                use_holidays=use_holidays,
                validation_seasonal_index_col=validation_seasonal_index_col,
                forward_seasonal_index_col=forward_seasonal_index_col,
                trend_col=trend_col,
                seasonality_col=seasonality_col,
            )
            for name, df in relevant_history_nas_filled.groupby(forecast_level)
        )

        logger.info("Collected results from parallel processing ...")

        # collect separate lists from the list of tuples returned by multiprocessing function
        all_model_validation_predictions = [x[0] for x in all_results]
        all_model_forecasts = [x[1] for x in all_results]
        all_model_desc_list = [x[2] for x in all_results]

        all_model_valid_df = concat_to_dataframe(all_model_validation_predictions)

        if all_model_valid_df.empty:
            logger.warning("Validation predictions empty for slice {}...".format(df_keys))
            return OutputForecast, OutputDescription

        all_model_forecasts_df = concat_to_dataframe(all_model_forecasts)

        if all_model_forecasts_df.empty:
            logger.warning("Forecasts empty for slice {}...".format(df_keys))
            return OutputForecast, OutputDescription

        all_model_desc_df = concat_to_dataframe(all_model_desc_list)
        all_model_desc_df.insert(0, version_col, input_version)

        ActualsAndForecastData = all_model_valid_df.append(
            all_model_forecasts_df, ignore_index=True
        )
        ActualsAndForecastData.insert(0, version_col, input_version)

        # collect the required columns
        req_data = TimeDimension[[relevant_time_name, partial_week_col, day_col]]

        # Calculate num of days in each partial week
        req_data[num_days_col] = req_data.groupby(partial_week_col)[day_col].transform("count")

        # Calculate num of days by fcst generation bucket
        req_data[total_num_days_col] = req_data.groupby(relevant_time_name)[day_col].transform(
            "count"
        )

        # Calculate the profile
        req_data[stat_bucket_weight_col] = req_data[num_days_col].divide(
            req_data[total_num_days_col]
        )

        # Collect all unique partial weeks
        PWProfile = req_data[
            [relevant_time_name, partial_week_col, stat_bucket_weight_col]
        ].drop_duplicates()

        StatBucketWeight = PWProfile[[partial_week_col, stat_bucket_weight_col]].drop_duplicates()

        # join with PW profile
        ActualsAndForecastData_at_pw = ActualsAndForecastData.merge(
            PWProfile, on=relevant_time_name, how="inner"
        )
        ActualsAndForecastData_at_pw.drop(relevant_time_name, axis=1, inplace=True)

        # multiply by weights to get values to PW
        cols_to_disagg = [
            x
            for x in ActualsAndForecastData_at_pw.columns
            if x not in forecast_level + [relevant_time_name, partial_week_col, version_col]
        ]

        for the_col in cols_to_disagg:
            ActualsAndForecastData_at_pw[the_col] = (
                ActualsAndForecastData_at_pw[the_col]
                * ActualsAndForecastData_at_pw[stat_bucket_weight_col]
            )

        # Collect data as required by best fit script
        bound_cols = [x for x in ActualsAndForecastData_at_pw.columns if "80%" in x]
        if bound_cols:
            ForecastBounds = ActualsAndForecastData_at_pw[
                forecast_level + [partial_week_col] + bound_cols
            ]
            ForecastBounds.insert(0, version_col, input_version)
            # join with stat bucket weight to disaggregate values to PW
            ForecastBounds = ForecastBounds.merge(
                StatBucketWeight, on=partial_week_col, how="inner"
            )
            ForecastBounds.drop([TimeLevel], axis=1, inplace=True)

        else:
            ForecastBounds = pd.DataFrame()

        ActualsAndForecastData_at_pw.drop(bound_cols, axis=1, inplace=True)

        ForecastGenTimeBucket = pd.DataFrame(
            {
                version_col: input_version,
                fcst_gen_time_bucket_col: fcst_gen_time_bucket,
            },
            index=[0],
        )

        AssignedAlgoList = relevant_history_nas_filled[
            [
                LocationLevel,
                ItemLevel,
                ChannelLevel,
                RegionLevel,
                AccountLevel,
                PnLLevel,
                DemandDomainLevel,
                assigned_algo_list_col,
            ]
        ]
        AssignedAlgoList.insert(0, version_col, input_version)

        grains_as_string = ",".join(
            [
                ItemLevel,
                LocationLevel,
                ChannelLevel,
                RegionLevel,
                AccountLevel,
                PnLLevel,
                DemandDomainLevel,
            ]
        )

        # for legacy best fit
        SelectionCriteria = pd.DataFrame(
            {
                version_col: input_version,
                bestfit_selection_criteria_col: "Validation Error",
            },
            index=[0],
        )

        AssignedAlgoList[o9Constants.ASSIGNED_RULE] = np.nan
        AssignedAlgoList[o9Constants.PLANNER_BESTFIT_ALGORITHM] = np.nan
        AssignedAlgoList[o9Constants.PLANNER_ASSIGNED_ALGORITHM_LIST] = np.nan
        AssignedAlgoList[o9Constants.SYSTEM_ASSIGNED_ALGORITHM_LIST] = AssignedAlgoList[
            assigned_algo_list_col
        ]

        # identify best fit model
        BestFitAlgo, ValidationError = identify_best_fit(
            Grains=grains_as_string,
            HistoryMeasure=ActualMeasure,
            TimeDimension=TimeDimension,
            ForecastParameters=ForecastParameters,
            CurrentTimePeriod=CurrentTimePeriod,
            ActualsAndForecastData=ActualsAndForecastData_at_pw,
            OverrideFlatLineForecasts="False",
            ForecastGenTimeBucket=ForecastGenTimeBucket,
            AssignedAlgoList=AssignedAlgoList,
            SelectionCriteria=SelectionCriteria,
            MasterAlgoList=MasterAlgoList,
            multiprocessing_num_cores=multiprocessing_num_cores,
            df_keys=df_keys,
        )

        if BestFitAlgo.empty:
            logger.warning("BestFitAlgo is empty ...")
            return OutputForecast, OutputDescription
        ForecastData = ActualsAndForecastData_at_pw.drop(ActualMeasure, axis=1)

        # populate planner best fit algo with same data
        BestFitAlgo[planner_best_fit_algo_col] = BestFitAlgo[system_best_fit_algo_col]

        # populate best fit predictions
        (
            BestFitForecast,
            BestFitAlgorithmCandidateOutput,
            BestFitViolationOutput,
        ) = populate_best_fit_forecast(
            Grains=",".join(
                [
                    ItemLevel,
                    LocationLevel,
                    ChannelLevel,
                    RegionLevel,
                    AccountLevel,
                    PnLLevel,
                    DemandDomainLevel,
                ]
            ),
            TimeDimension=TimeDimension,
            ForecastParameters=ForecastParameters,
            CurrentTimePeriod=CurrentTimePeriod,
            ForecastData=ForecastData,
            ForecastBounds=ForecastBounds,
            BestFitAlgo=BestFitAlgo,
            ForecastGenTimeBucket=ForecastGenTimeBucket,
            StatBucketWeight=StatBucketWeight,
            df_keys=df_keys,
        )

        BestFitForecast.rename(
            columns={system_stat_fcst_l1_col: stat_fcst_best_fit_col},
            inplace=True,
        )

        BestFitForecast = BestFitForecast.merge(
            ForecastData.drop(version_col, axis=1),
            on=forecast_level + [partial_week_col],
            how="inner",
        )

        BestFitForecast.drop(
            [
                stat_bucket_weight_col,
                system_stat_fcst_l1_ub_col,
                system_stat_fcst_l1_lb_col,
            ],
            axis=1,
            inplace=True,
        )

        forecast_period_partial_weeks = list(BestFitForecast[partial_week_col].unique())

        last_partial_week = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            partial_week_col,
            partial_week_key_col,
        )
        # get data for last one year - assume 65 partial weeks in a year
        last_one_year_pw = get_n_time_periods(
            last_partial_week,
            -65,
            TimeDimension[[partial_week_col, partial_week_key_col]].drop_duplicates(),
            {partial_week_col: partial_week_key_col},
            include_latest_value=True,
        )

        # extend last 12 months list to match future period length for mapping
        last_12_months = last_one_year_pw * math.ceil(
            len(forecast_period_partial_weeks) / len(last_one_year_pw)
        )
        last_12_months = last_12_months[: len(forecast_period_partial_weeks)]

        past_to_future_date_mappings = pd.DataFrame(
            {
                partial_week_col: last_12_months,
                future_periods_col: forecast_period_partial_weeks,
            }
        )

        # get last 12 months of data and map it with future periods for disaggregation
        forecast_profile = Input_Cleansed_History.merge(
            past_to_future_date_mappings, how="inner", on=partial_week_col
        )

        forecast_profile.drop(columns=[partial_week_col, version_col], inplace=True)

        forecast_profile.rename(columns={future_periods_col: partial_week_col}, inplace=True)

        if ItemLevel not in forecast_profile.columns:
            forecast_profile = forecast_profile.merge(
                Input_Attribute_PlanningItem[[L0ItemLevel, ItemLevel]].drop_duplicates(),
                on=L0ItemLevel,
                how="inner",
            )

        if LocationLevel not in forecast_profile.columns:
            forecast_profile = forecast_profile.merge(
                Input_Attribute_Location[[L0LocationLevel, LocationLevel]].drop_duplicates(),
                on=L0LocationLevel,
                how="inner",
            )

        if ChannelLevel not in forecast_profile.columns:
            forecast_profile = forecast_profile.merge(
                Input_Attribute_Channel[[L0ChannelLevel, ChannelLevel]].drop_duplicates(),
                on=L0ChannelLevel,
                how="inner",
            )

        if RegionLevel not in forecast_profile.columns:
            forecast_profile = forecast_profile.merge(
                Input_Attribute_Region[[L0RegionLevel, RegionLevel]].drop_duplicates(),
                on=L0RegionLevel,
                how="inner",
            )

        if AccountLevel not in forecast_profile.columns:
            forecast_profile = forecast_profile.merge(
                Input_Attribute_Account[[L0AccountLevel, AccountLevel]].drop_duplicates(),
                on=L0AccountLevel,
                how="inner",
            )

        if PnLLevel not in forecast_profile.columns:
            forecast_profile = forecast_profile.merge(
                Input_Attribute_PnL[[L0PnLLevel, PnLLevel]].drop_duplicates(),
                on=L0PnLLevel,
                how="inner",
            )

        if DemandDomainLevel not in forecast_profile.columns:
            forecast_profile = forecast_profile.merge(
                Input_Attribute_DemandDomain[
                    [L0DemandDomainLevel, DemandDomainLevel]
                ].drop_duplicates(),
                on=L0DemandDomainLevel,
                how="inner",
            )

        BestFitForecast.rename(columns=rename_mapping, inplace=True)

        # Combined profile with best fit forecast
        combined_df = BestFitForecast.merge(
            forecast_profile,
            on=interactive_stat_level + [partial_week_col],
            how="inner",
        )

        # calculate the split percentage for each Stat Item based on Interactive Stat Level
        combined_df[group_sum_col] = combined_df.groupby(
            interactive_stat_level + [partial_week_col]
        )[ActualMeasure].transform("sum")

        logger.info("Calculating split percentages ...")
        combined_df[split_percent_col] = combined_df[ActualMeasure].divide(
            combined_df[group_sum_col]
        )

        # fill NAs which come out of division by zero
        combined_df[split_percent_col].fillna(0, inplace=True)

        required_fcst_cols = [x for x in BestFitForecast.columns if "Forecast" in x or "Fcst" in x]

        for the_col in required_fcst_cols:
            combined_df[the_col] = combined_df[the_col] * combined_df[split_percent_col]

        # Rename and Output formatting
        req_cols = [
            L0ItemLevel,
            L0LocationLevel,
            L0ChannelLevel,
            L0RegionLevel,
            L0AccountLevel,
            L0PnLLevel,
            L0DemandDomainLevel,
            partial_week_col,
        ] + required_fcst_cols
        OutputForecast = combined_df[req_cols].drop_duplicates()
        OutputForecast.insert(0, version_col, input_version)

        OutputForecast.rename(columns=rename_mapping, inplace=True)
        OutputForecast = OutputForecast[OutputForecast_cols]

        # create a dimension filter list
        filter_list = [
            L0ItemLevel,
            L0LocationLevel,
            L0ChannelLevel,
            L0RegionLevel,
            L0AccountLevel,
            L0PnLLevel,
            L0DemandDomainLevel,
        ] + interactive_stat_level

        dim_filter_list = list(set(filter_list))
        # dim_filter_list = [dim_filter_list.append(x) for x in filter_list if x not in dim_filter_list]
        logger.info(dim_filter_list)
        all_model_desc_df = all_model_desc_df.merge(
            BestFitAlgo, on=forecast_level + [version_col], how="inner"
        )
        all_model_desc_df_copy = all_model_desc_df[
            all_model_desc_df[stat_algo_col] == all_model_desc_df[system_best_fit_algo_col]
        ]
        all_model_desc_df_copy[stat_algo_col] = "Bestfit Model"
        all_model_parameters = concat_to_dataframe([all_model_desc_df, all_model_desc_df_copy])

        ForecastModelAndBestFitAlgo = all_model_parameters.pivot_table(
            algorithm_parameters_col,
            forecast_level,
            stat_algo_col,
            aggfunc="first",
        ).reset_index()

        ForecastModelAndBestFitAlgo_item_loc_sd = ForecastModelAndBestFitAlgo.merge(
            forecast_profile[dim_filter_list].drop_duplicates(),
            on=interactive_stat_level,
            how="inner",
        )

        model_cols = [x for x in ForecastModelAndBestFitAlgo.columns if "Model" in x]
        OutputDescription = pd.melt(
            ForecastModelAndBestFitAlgo_item_loc_sd,
            id_vars=[
                L0ItemLevel,
                L0LocationLevel,
                L0ChannelLevel,
                L0RegionLevel,
                L0AccountLevel,
                L0PnLLevel,
                L0DemandDomainLevel,
            ],
            value_vars=model_cols,
            var_name=stat_model_col,
            value_name=stat_fcst_desc_col,
        )

        rename_mapping_model["Bestfit Model"] = "Best Fit"
        OutputDescription[stat_model_col] = OutputDescription[stat_model_col].map(
            rename_mapping_model
        )

        OutputDescription.insert(0, version_col, input_version)

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        OutputForecast = pd.DataFrame(columns=OutputForecast_cols)
        OutputDescription = pd.DataFrame(columns=OutputDescription_cols)

    return OutputForecast, OutputDescription
