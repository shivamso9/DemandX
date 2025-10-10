import logging

import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.common_utils import (
    get_last_time_period,
    get_n_time_periods,
    get_seasonal_periods,
)
from o9Reference.common_utils.data_utils import validate_output
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.disaggregate_data import disaggregate_data

from helpers.disaggregation import join_lowest_level
from helpers.outlier_correction import cleanse_data_wrapper

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

col_mapping = {"Stat Cleansed History System": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Input_History,
    Outlier_Parameters,
    Input_Attribute_PlanningItem,
    Input_Attribute_Location,
    TimeDimension,
    Input_Stat_Level,
    CurrentTimePeriod,
    PartialWeekMapping,
    Input_Attribute_Channel,
    Input_Attribute_Region,
    Input_Attribute_Account,
    Input_Attribute_PnL,
    Input_Attribute_DemandDomain,
    multiprocessing_num_cores=4,
    df_keys={},
):
    plugin_name = "DP002StatCleansing"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    version_col = "Version.[Version Name]"
    L0ItemLevel = "Item.[Planning Item]"
    L0LocationLevel = "Location.[Planning Location]"
    L0ChannelLevel = "Channel.[Planning Channel]"
    L0RegionLevel = "Region.[Planning Region]"
    L0AccountLevel = "Account.[Planning Account]"
    L0PnLLevel = "PnL.[Planning PnL]"
    L0DemandDomainLevel = "Demand Domain.[Planning Demand Domain]"
    L0TimeLevel = "Time.[Week]"
    ActualMeasure = "Stat Raw History"
    partial_week_col = "Time.[Partial Week]"
    week_col = "Time.[Week]"
    week_key_col = "Time.[WeekKey]"
    month_col = "Time.[Month]"
    month_key_col = "Time.[MonthKey]"
    planning_month_col = "Time.[Planning Month]"
    planning_month_key_col = "Time.[PlanningMonthKey]"
    quarter_col = "Time.[Quarter]"
    quarter_key_col = "Time.[QuarterKey]"
    planning_quarter_col = "Time.[Planning Quarter]"
    planning_quarter_key_col = "Time.[PlanningQuarterKey]"

    logger.info("multiprocessing_num_cores : {}".format(multiprocessing_num_cores))

    # Outlier Parameters
    upper_threshold_col = "Stat History Outlier Upper Threshold Limit"
    lower_threshold_col = "Stat History Outlier Lower Threshold Limit"
    outlier_correction_col = "Stat History Outlier Correction"
    outlier_method_col = "Stat History Outlier Method"

    # output columns - output measures
    cleansed_data_col = "Stat Cleansed History System"
    upper_bound_col = "Outlier Upper Threshold"
    lower_bound_col = "Outlier Lower Threshold"
    actual_mean_col = "Actual Median"

    split_percent_col = "Split Percent"
    group_sum_col = "Group Sum"

    # pw disaggregation
    time_week_col = "Time.[Week]"
    time_partial_week_col = "Time.[Partial Week]"
    is_day_col = "Is Day"
    split_ratio_col = "Split Ratio"

    cols_required_in_output = [
        version_col,
        L0LocationLevel,
        L0ItemLevel,
        L0ChannelLevel,
        L0RegionLevel,
        L0AccountLevel,
        L0PnLLevel,
        L0DemandDomainLevel,
        partial_week_col,
        cleansed_data_col,
    ]
    Output_Cleansed_History = pd.DataFrame(columns=cols_required_in_output)
    try:

        if PartialWeekMapping.empty:
            logger.warning(
                "PartialWeekMapping input is empty, please populate the measure 'Is Day' and execute the plugin again ..."
            )
            return Output_Cleansed_History

        if Input_Stat_Level.empty:
            logger.warning(
                "Input_Stat_Level input is empty, please set up the attributes from 'Interactive Outlier Setup' and execute the plugin again ..."
            )
            return Output_Cleansed_History

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

        # Default is week
        frequency = "Weekly"
        forecast_periods = 104
        validation_periods = 26
        history_periods = 52 * 3
        relevant_time_name = week_col
        relevant_time_key = week_key_col
        relevant_time_cols = [partial_week_col, week_col, week_key_col]

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
        elif TimeLevel == "Time.[Month]":
            frequency = "Monthly"
            forecast_periods = 24
            validation_periods = 6
            history_periods = 12 * 3
            relevant_time_cols = [partial_week_col, month_col, month_key_col]
            relevant_time_name = month_col
            relevant_time_key = month_key_col
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
            df=Input_History,
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

        # get last time period name
        last_time_period_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )
        input_version = Input_History[version_col].iloc[0]

        logger.info("last_time_period_name : {}".format(last_time_period_name))

        time_series_freq = get_seasonal_periods(frequency)

        last_n_periods = get_n_time_periods(
            latest_value=last_time_period_name,
            periods=-history_periods,
            time_mapping=relevant_time_mapping,
            time_attribute=time_attribute_dict,
            include_latest_value=True,
        )

        logger.info("last_n_periods : {}".format(last_n_periods))

        cleanse_level = [
            ItemLevel,
            LocationLevel,
            ChannelLevel,
            RegionLevel,
            AccountLevel,
            PnLLevel,
            DemandDomainLevel,
        ]
        logger.info("cleanse_level : {}".format(cleanse_level))

        logger.info("Filling missing dates ...")

        # Copy TimeLevel into relevant_time_name to make fill missing dates work
        aggregated_history[relevant_time_name] = aggregated_history[TimeLevel]

        # fill missing dates
        relevant_history_nas_filled = fill_missing_dates(
            actual=aggregated_history,
            forecast_level=cleanse_level,
            time_mapping=relevant_time_mapping,
            history_measure=ActualMeasure,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            relevant_time_periods=last_n_periods,
            fill_nulls_with_zero=True,
        )

        # sort values
        relevant_history_nas_filled.sort_values(cleanse_level + [relevant_time_key], inplace=True)

        logger.info("Performing outlier correction for all intersections ...")
        all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(cleanse_data_wrapper)(
                group,
                cleanse_level,
                relevant_time_key,
                ActualMeasure,
                cleansed_data_col,
                upper_bound_col,
                lower_bound_col,
                actual_mean_col,
                TimeLevel,
                Outlier_Parameters,
                upper_threshold_col,
                lower_threshold_col,
                outlier_method_col,
                outlier_correction_col,
                time_series_freq,
                interactive_stat=True,
            )
            for name, group in relevant_history_nas_filled.groupby(cleanse_level)
        )

        logger.info("Collected results from parallel processing ...")

        # Concatenate all results to one dataframe
        cleansed_data = concat_to_dataframe(all_results)

        # validate if output dataframe contains result for all groups present in input
        validate_output(
            input_df=relevant_history_nas_filled,
            output_df=cleansed_data,
            forecast_level=cleanse_level,
        )

        if len(cleansed_data) == 0:
            logger.warning(
                "No records after processing for slice : {}, returning empty dataframe".format(
                    df_keys
                )
            )
            return Output_Cleansed_History

        validate_output(
            input_df=relevant_history_nas_filled,
            output_df=cleansed_data,
            forecast_level=cleanse_level,
        )

        logger.info("Generating time attributes for {}".format(L0TimeLevel))

        # Get time related attributes for L0Level
        frequency = "Weekly"
        relevant_time_name = "Time.[Week]"
        relevant_time_key = "Time.[WeekKey]"
        history_period = 156

        L0_last_time_period = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            L0TimeLevel,
            week_key_col,
        )
        L0_time_attribute_dict = {week_col: week_key_col}

        L0_time_mapping = TimeDimension[[week_col, week_key_col]].drop_duplicates()

        time_series_freq = get_seasonal_periods(frequency)

        L0_last_n_periods = get_n_time_periods(
            latest_value=L0_last_time_period,
            periods=-history_period,
            time_mapping=L0_time_mapping,
            time_attribute=L0_time_attribute_dict,
            include_latest_value=True,
        )

        cleanse_level = [
            L0ItemLevel,
            L0LocationLevel,
            L0ChannelLevel,
            L0RegionLevel,
            L0AccountLevel,
            L0PnLLevel,
            L0DemandDomainLevel,
        ]

        logger.info("cleanse_level : {}".format(cleanse_level))

        logger.info("Filling missing dates ...")

        L0_Input_History = Input_History.merge(
            TimeDimension[[partial_week_col, week_col, week_key_col]].drop_duplicates(),
            on=partial_week_col,
            how="inner",
        )

        L0_Input_History = (
            L0_Input_History.groupby(cleanse_level + [L0TimeLevel])
            .sum()[[ActualMeasure]]
            .reset_index()
        )

        # fill missing dates
        L0_relevant_history_nas_filled = fill_missing_dates(
            actual=L0_Input_History,
            forecast_level=cleanse_level,
            time_mapping=L0_time_mapping,
            history_measure=ActualMeasure,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            relevant_time_periods=L0_last_n_periods,
            fill_nulls_with_zero=True,
        )

        # perform outlier correction at L0 level with fixed sigma to generate profile
        L0_Outlier_Parameters = Outlier_Parameters.copy(deep=True)
        L0_Outlier_Parameters[outlier_correction_col] = "Limit"
        L0_Outlier_Parameters[lower_threshold_col] = 3.0
        L0_Outlier_Parameters[upper_threshold_col] = 3.0
        L0_Outlier_Parameters[outlier_method_col] = "Fixed Sigma"

        logger.info("L0_Outlier_Parameters \n{}".format(L0_Outlier_Parameters))

        logger.info("Performing outlier correction for all intersections ...")

        L0_all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(cleanse_data_wrapper)(
                group,
                cleanse_level,
                relevant_time_key,
                ActualMeasure,
                cleansed_data_col,
                upper_bound_col,
                lower_bound_col,
                actual_mean_col,
                L0TimeLevel,
                L0_Outlier_Parameters,
                upper_threshold_col,
                lower_threshold_col,
                outlier_method_col,
                outlier_correction_col,
                time_series_freq,
                interactive_stat=True,
            )
            for name, group in L0_relevant_history_nas_filled.groupby(cleanse_level)
        )

        logger.info("Collected results from parallel processing ...")
        # Concatenate all results to one dataframe
        L0_cleansed_data = concat_to_dataframe(L0_all_results)

        # validate if output dataframe contains result for all groups present in input
        validate_output(
            input_df=L0_relevant_history_nas_filled,
            output_df=L0_cleansed_data,
            forecast_level=cleanse_level,
        )

        logger.info("Joining with dimension master ...")

        # duplicate column for better understanding
        L0_cleansed_data_col = "L0_" + cleansed_data_col
        L0_cleansed_data.rename(columns={cleansed_data_col: L0_cleansed_data_col}, inplace=True)

        if ItemLevel not in L0_cleansed_data.columns:
            L0_cleansed_data = L0_cleansed_data.merge(
                Input_Attribute_PlanningItem[[L0ItemLevel, ItemLevel]].drop_duplicates(),
                on=L0ItemLevel,
                how="inner",
            )

        if LocationLevel not in L0_cleansed_data.columns:
            L0_cleansed_data = L0_cleansed_data.merge(
                Input_Attribute_Location[[L0LocationLevel, LocationLevel]].drop_duplicates(),
                on=L0LocationLevel,
                how="inner",
            )

        if ChannelLevel not in L0_cleansed_data.columns:
            L0_cleansed_data = L0_cleansed_data.merge(
                Input_Attribute_Channel[[L0ChannelLevel, ChannelLevel]].drop_duplicates(),
                on=L0ChannelLevel,
                how="inner",
            )

        if RegionLevel not in L0_cleansed_data.columns:
            L0_cleansed_data = L0_cleansed_data.merge(
                Input_Attribute_Region[[L0RegionLevel, RegionLevel]].drop_duplicates(),
                on=L0RegionLevel,
                how="inner",
            )

        if AccountLevel not in L0_cleansed_data.columns:
            L0_cleansed_data = L0_cleansed_data.merge(
                Input_Attribute_Account[[L0AccountLevel, AccountLevel]].drop_duplicates(),
                on=L0AccountLevel,
                how="inner",
            )

        if PnLLevel not in L0_cleansed_data.columns:
            L0_cleansed_data = L0_cleansed_data.merge(
                Input_Attribute_PnL[[L0PnLLevel, PnLLevel]].drop_duplicates(),
                on=L0PnLLevel,
                how="inner",
            )

        if DemandDomainLevel not in L0_cleansed_data.columns:
            L0_cleansed_data = L0_cleansed_data.merge(
                Input_Attribute_DemandDomain[
                    [L0DemandDomainLevel, DemandDomainLevel]
                ].drop_duplicates(),
                on=L0DemandDomainLevel,
                how="inner",
            )

        if TimeLevel not in L0_cleansed_data.columns:
            L0_cleansed_data = L0_cleansed_data.merge(
                TimeDimension[[L0TimeLevel, TimeLevel]].drop_duplicates(),
                on=L0TimeLevel,
                how="inner",
            )

        higher_cleansed_col = "Hi " + cleansed_data_col
        # rename existing columns on higher level dataframe
        cleansed_data.rename(columns={cleansed_data_col: higher_cleansed_col}, inplace=True)

        # drop irrelevant values at L0 level
        L0_cleansed_data.drop(
            [lower_bound_col, upper_bound_col, actual_mean_col],
            axis=1,
            inplace=True,
        )

        # Join with numbers generated on higher grain
        disaggregated_cleansed_data = L0_cleansed_data.merge(
            cleansed_data, on=interactive_stat_level + [TimeLevel]
        )

        logger.info("Disaggregating values ...")

        # take group sum at higher level
        disaggregated_cleansed_data[group_sum_col] = disaggregated_cleansed_data.groupby(
            interactive_stat_level + [TimeLevel]
        )[L0_cleansed_data_col].transform("sum")

        logger.info("Calculating split percentages ...")
        # divide L0 numbers by group sum to get proportions
        disaggregated_cleansed_data[split_percent_col] = disaggregated_cleansed_data[
            L0_cleansed_data_col
        ].divide(disaggregated_cleansed_data[group_sum_col])

        # Fill NAs yielded due to division by zero
        disaggregated_cleansed_data[split_percent_col].fillna(0, inplace=True)

        # multiply higher level number with ratio to get it down to lower level
        disaggregated_cleansed_data[cleansed_data_col] = (
            disaggregated_cleansed_data[split_percent_col]
            * disaggregated_cleansed_data[higher_cleansed_col]
        )

        logger.info("Values disaggregated to lowest level ...")

        logger.info("Selecting the required columns ...")
        req_cols = lowest_level + [L0TimeLevel, cleansed_data_col]
        Output_Cleansed_History = disaggregated_cleansed_data[req_cols]

        Output_Cleansed_History.insert(0, version_col, input_version)

        logger.info("Disaggregating to partial week ...")

        # select required cols from mapping
        PartialWeekMapping = PartialWeekMapping[[time_week_col, time_partial_week_col, is_day_col]]

        # calculate split ratio
        PartialWeekMapping[split_ratio_col] = PartialWeekMapping[is_day_col].divide(
            PartialWeekMapping.groupby(time_week_col)[is_day_col].transform(sum)
        )

        # perform disaggregation
        Output_Cleansed_History = disaggregate_data(
            source_df=Output_Cleansed_History,
            source_grain=time_week_col,
            target_grain=time_partial_week_col,
            profile_df=PartialWeekMapping,
            profile_col=split_ratio_col,
            cols_to_disaggregate=[cleansed_data_col],
        )

        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        Output_Cleansed_History = pd.DataFrame(columns=cols_required_in_output)

    return Output_Cleansed_History
