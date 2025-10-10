"""DP049PopulateNPIFcst module.

Populate NPI Forecast.
"""

import logging

import numpy as np
import pandas as pd
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

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def get_relevant_ramp_up_profile(
    DefaultProfile,
    plc_profile_col,
    npi_profile,
    npi_profile_bucket,
    lifecycle_bucket_col,
    default_profile_col,
    profile_bucket_col,
) -> pd.DataFrame:
    """Get relevant ramp up profile."""
    relevant_ramp_up_profile = pd.DataFrame()
    filter_clause = (DefaultProfile[plc_profile_col] == npi_profile) & (
        DefaultProfile[profile_bucket_col] == npi_profile_bucket
    )
    relevant_ramp_up_profile = DefaultProfile[filter_clause]

    if relevant_ramp_up_profile.empty:
        pass
    else:
        relevant_ramp_up_profile = relevant_ramp_up_profile[
            [lifecycle_bucket_col, default_profile_col]
        ].drop_duplicates()
    return relevant_ramp_up_profile


def create_relevant_rank_column(bucket: str) -> str:
    """Create relevant rank column."""
    relevant_rank_column = "Time.[" + bucket + "Key]"
    relevant_rank_column = relevant_rank_column.replace(" ", "")
    return relevant_rank_column


col_mapping = {
    "NPI Fcst": float,
    "NPI Profile L0": float,
    "NPI Profile L0 Normalized": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    forecast_data,
    parameter_data,
    Grains,
    df_keys,
    TimeDimension,
    stat_bucket_weight,
    DefaultProfile,
):
    """Run the function to populate NPI Forecast.

    Return NPI Forecast data frame.
    """
    plugin_name = "DP049PopulateNPIFcst"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"

    partial_week_col = "Time.[Partial Week]"
    pl_month_key_col = "Time.[PlanningMonthKey]"
    month_key_col = "Time.[MonthKey]"
    week_key_col = "Time.[WeekKey]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    day_key_col = "Time.[DayKey]"

    intro_date_col = "Intro Date"
    disco_date_col = "Disco Date"
    ramp_up_period_col = "User Defined Ramp Up Period"
    ramp_up_volume_col = "User Defined Ramp Up Volume"
    scaling_factor_col = "Scaling Factor"
    initial_build_col = "Initial Build"
    like_item_forecast_col = "Like Item Fcst"
    default_profile_col = "Default Profile"
    lifecycle_bucket_col = "Lifecycle Time.[Lifecycle Bucket]"
    lifecycle_bucket_key_col = "Lifecycle Time.[LifecycleBucketKey]"
    npi_profile_col = "NPI Profile"
    npi_profile_bucket_col = "NPI Profile Bucket"
    plc_profile_col = "PLC Profile.[PLC Profile]"
    profile_bucket_col = "PLC Profile.[PLC Time Bucket]"

    is_ramp_up_period_col = "Is Ramp Up Period"
    stat_bucket_weight_col = "Num Days"
    sum_of_stat_bucket_weight_col = "Sum of Stat Bucket Weight"
    disagg_ramp_up_volume_col = "Disaggregated Ramp up Volume"
    disagg_initial_build_col = "Disagg Initial Build"
    normalized_stat_bucket_weight_col = "Normalized Stat Bucket Weight"

    # output measure
    output_measure = "NPI Fcst"
    ramp_up_profile_disagg_to_PW_col = "NPI Profile L0"
    normalized_ramp_up_profile_disagg_to_PW_col = "NPI Profile L0 Normalized"

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get granular level
    dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
    logger.info("dimensions : {} ...".format(dimensions))

    assert len(dimensions) > 0, "dimensions cannot be empty ..."

    all_output_measures = [
        output_measure,
        ramp_up_profile_disagg_to_PW_col,
        normalized_ramp_up_profile_disagg_to_PW_col,
    ]

    cols_required_in_output = [version_col, partial_week_col] + dimensions + all_output_measures

    try:
        assert len(TimeDimension) > 0, "time dimension cannot be empty ..."

        # if intro date is missing, ignore that combination
        parameter_data = parameter_data[parameter_data[intro_date_col].notna()]

        if len(parameter_data) == 0:
            logger.warning(
                "Input is None/Empty for slice : {}, check input Parameters".format(df_keys)
            )
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=cols_required_in_output)

        input_version = parameter_data[version_col].iloc[0]
        logger.debug(f"input_version : {input_version}")

        # default ramp up period is 12
        parameter_data[ramp_up_period_col] = parameter_data[ramp_up_period_col].fillna(12)

        # cap negatives to zero
        parameter_data[ramp_up_period_col] = np.where(
            parameter_data[ramp_up_period_col] < 0,
            0,
            parameter_data[ramp_up_period_col],
        )

        # setting null ramp up volume to 0
        parameter_data[ramp_up_volume_col] = parameter_data[ramp_up_volume_col].fillna(0)

        # cap negatives to zero
        parameter_data[ramp_up_volume_col] = np.where(
            parameter_data[ramp_up_volume_col] < 0,
            0,
            parameter_data[ramp_up_volume_col],
        )

        # default scaling factor is 1
        parameter_data[scaling_factor_col] = parameter_data[scaling_factor_col].fillna(1)
        # if scaling factor is less than or equal to zero, change it 1.0
        parameter_data[scaling_factor_col] = np.where(
            parameter_data[scaling_factor_col] <= 0,
            1,
            parameter_data[scaling_factor_col],
        )

        # setting null initial build to 0
        parameter_data[initial_build_col] = parameter_data[initial_build_col].fillna(0)
        # cap negatives to zero
        parameter_data[initial_build_col] = np.where(
            parameter_data[initial_build_col] < 0,
            0,
            parameter_data[initial_build_col],
        )

        intro_time_key_name = "IntroDate"
        disco_time_key_name = "DiscoDate"

        # if DefaultProfile.empty:
        #     logger.warning(
        #         "DefaultProfile is empty for slice : {}".format(df_keys)
        #     )
        #     return pd.DataFrame(columns=cols_required_in_output)

        DefaultProfile[lifecycle_bucket_key_col] = pd.to_datetime(
            DefaultProfile[lifecycle_bucket_key_col],
            infer_datetime_format=True,
        )

        DefaultProfile[lifecycle_bucket_col] = DefaultProfile[lifecycle_bucket_key_col].rank(
            method="dense"
        )

        DefaultProfile[lifecycle_bucket_col] = DefaultProfile[lifecycle_bucket_col].astype(int)

        DefaultProfile[default_profile_col] = DefaultProfile[default_profile_col].astype(float)

        # collect datetime key columns
        key_cols = [
            pl_month_key_col,
            month_key_col,
            week_key_col,
            partial_week_key_col,
            day_key_col,
        ]
        logger.info("Converting key cols to datetime format ...")

        # convert to datetime
        TimeDimension[key_cols] = TimeDimension[key_cols].apply(
            pd.to_datetime, infer_datetime_format=True
        )
        # collect available min/max from TimeDimension
        min_possible_date = np.datetime64(TimeDimension[day_key_col].min())
        max_possible_date = np.datetime64(TimeDimension[day_key_col].max())

        parameter_data[intro_date_col] = pd.to_datetime(
            parameter_data[intro_date_col],
            infer_datetime_format=True,
        )
        # fill nas in disco date with max possible date in time master
        parameter_data[disco_date_col].fillna(max_possible_date, inplace=True)

        parameter_data[disco_date_col] = pd.to_datetime(
            parameter_data[disco_date_col],
            infer_datetime_format=True,
        )

        # remove time component if any
        parameter_data[intro_date_col] = parameter_data[intro_date_col].dt.normalize()
        parameter_data[disco_date_col] = parameter_data[disco_date_col].dt.normalize()

        # drop version col
        stat_bucket_weight.drop(version_col, axis=1, inplace=True)

        # override user provided values - we cannot populate any data beyond the min/max value in time dimension
        parameter_data[intro_date_col] = np.clip(
            parameter_data[intro_date_col],
            min_possible_date,
            max_possible_date,
        )
        parameter_data[disco_date_col] = np.clip(
            parameter_data[disco_date_col],
            min_possible_date,
            max_possible_date,
        )

        day_to_partial_week_key_df = TimeDimension[
            [day_key_col, partial_week_key_col]
        ].drop_duplicates()

        parameter_data = parameter_data.merge(
            day_to_partial_week_key_df,
            left_on=intro_date_col,
            right_on=day_key_col,
            how="inner",
        )
        parameter_data.drop(day_key_col, axis=1, inplace=True)
        parameter_data.rename(
            columns={partial_week_key_col: intro_time_key_name},
            inplace=True,
        )

        parameter_data = parameter_data.merge(
            day_to_partial_week_key_df,
            left_on=disco_date_col,
            right_on=day_key_col,
            how="inner",
        )
        parameter_data.drop(day_key_col, axis=1, inplace=True)
        parameter_data.rename(
            columns={partial_week_key_col: disco_time_key_name},
            inplace=True,
        )

        col_list = [
            partial_week_col,
            partial_week_key_col,
            week_key_col,
            month_key_col,
            pl_month_key_col,
        ]
        generation_time_mapping = TimeDimension[col_list].drop_duplicates()

        pw_master = TimeDimension[[partial_week_col, partial_week_key_col]].drop_duplicates()

        # getting partial weeks which lies in between corresponding intro and disco dates, get ramp up profile
        master_data_list = []

        for _, the_group in parameter_data.groupby(dimensions):
            try:
                # in case include null rows is not set to False, there could be more than one row in a group
                the_group.drop_duplicates(inplace=True)
                if len(the_group) > 1:
                    raise AssertionError("duplicates found in parameter data for the intersection")

                intro_key = the_group[intro_time_key_name].iloc[0]
                disco_key = the_group[disco_time_key_name].iloc[0]
                ramp_up_volume = the_group[ramp_up_volume_col].iloc[0]
                npi_profile = str(the_group[npi_profile_col].iloc[0])
                npi_profile_bucket = str(the_group[npi_profile_bucket_col].iloc[0])

                filter_clause = (pw_master[partial_week_key_col] >= intro_key) & (
                    pw_master[partial_week_key_col] <= disco_key
                )
                the_time_mapping = pw_master[filter_clause].drop_duplicates()

                # merge the parameter data with time mapping
                the_data = create_cartesian_product(df1=the_group, df2=the_time_mapping)

                # merge master data with generation time mapping to get relevant key cols
                the_data = the_data.merge(
                    generation_time_mapping,
                    on=[partial_week_col, partial_week_key_col],
                )

                # profile not availabe - NPI Fcst will be calculated, profile and profile normalized will be null
                if (
                    npi_profile == "None"
                    or npi_profile == "nan"
                    or npi_profile == ""
                    or npi_profile_bucket == "None"
                    or npi_profile_bucket == "nan"
                    or npi_profile_bucket == ""
                    or ramp_up_volume == 0
                ):
                    # retain only relevant columns
                    req_cols = (
                        [version_col]
                        + dimensions
                        + [
                            partial_week_col,
                            partial_week_key_col,
                            ramp_up_period_col,
                            ramp_up_volume_col,
                            scaling_factor_col,
                            initial_build_col,
                            npi_profile_col,
                            npi_profile_bucket_col,
                        ]
                    )
                    the_data_at_relevant_grain = the_data[req_cols].drop_duplicates()

                    relevant_forecast_data = forecast_data.merge(
                        the_group[dimensions], on=dimensions, how="inner"
                    )
                    if relevant_forecast_data.empty:
                        logger.warning("Forecast data not available for the intersection")
                        continue

                    # left join : NPI Fcst might or might not be there for the first bucket
                    the_data_at_relevant_grain = the_data_at_relevant_grain.merge(
                        relevant_forecast_data,
                        on=[version_col] + dimensions + [partial_week_col],
                        how="left",
                    )

                    # populate npi fcst
                    the_data_at_relevant_grain[output_measure] = np.where(
                        the_data_at_relevant_grain[like_item_forecast_col].notna(),
                        the_data_at_relevant_grain[like_item_forecast_col]
                        * the_data_at_relevant_grain[scaling_factor_col],
                        np.nan,
                    )

                    # getting first partial week
                    first_pw = TimeDimension[TimeDimension[partial_week_key_col] <= intro_key][
                        partial_week_key_col
                    ].iloc[-1]

                    the_data_at_relevant_grain.loc[
                        the_data_at_relevant_grain[partial_week_key_col] == first_pw,
                        output_measure,
                    ] = the_data_at_relevant_grain[output_measure].add(
                        the_data_at_relevant_grain[initial_build_col],
                        fill_value=0,
                    )

                    # profile l0 and profile l0 normalized will be null
                    the_data_at_relevant_grain[ramp_up_profile_disagg_to_PW_col] = np.nan
                    the_data_at_relevant_grain[normalized_ramp_up_profile_disagg_to_PW_col] = np.nan

                    # select relevant columns
                    the_result = the_data_at_relevant_grain[
                        cols_required_in_output
                    ].drop_duplicates()
                else:
                    relevant_rank_column = create_relevant_rank_column(bucket=npi_profile_bucket)

                    # retain only relevant columns
                    req_cols = (
                        [version_col]
                        + dimensions
                        + [
                            relevant_rank_column,
                            ramp_up_period_col,
                            ramp_up_volume_col,
                            scaling_factor_col,
                            initial_build_col,
                            npi_profile_col,
                            npi_profile_bucket_col,
                        ]
                    )
                    the_data_at_relevant_grain = the_data[req_cols].drop_duplicates()

                    the_data_at_relevant_grain[lifecycle_bucket_col] = the_data_at_relevant_grain[
                        relevant_rank_column
                    ].rank(axis=0, method="dense", ascending=True)
                    relevant_ramp_up_profile = get_relevant_ramp_up_profile(
                        DefaultProfile,
                        plc_profile_col,
                        npi_profile,
                        npi_profile_bucket,
                        lifecycle_bucket_col,
                        default_profile_col,
                        profile_bucket_col,
                    )

                    # left join is required to retain all dates in future beyond the ramp up period
                    the_data_at_relevant_grain = the_data_at_relevant_grain.merge(
                        relevant_ramp_up_profile,
                        on=lifecycle_bucket_col,
                        how="left",
                    )

                    # mark ramp up period
                    the_data_at_relevant_grain[is_ramp_up_period_col] = np.where(
                        the_data_at_relevant_grain[ramp_up_period_col]
                        >= the_data_at_relevant_grain[lifecycle_bucket_col],
                        1,
                        0,
                    )

                    # add the relevant rank column to stat bucket weight
                    the_stat_bucket_weight = stat_bucket_weight.merge(
                        TimeDimension[
                            [
                                partial_week_col,
                                relevant_rank_column,
                                partial_week_key_col,
                            ]
                        ].drop_duplicates(),
                        on=partial_week_col,
                        how="inner",
                    )

                    # normalize the stat bucket weight
                    the_stat_bucket_weight[sum_of_stat_bucket_weight_col] = (
                        the_stat_bucket_weight.groupby(relevant_rank_column)[
                            stat_bucket_weight_col
                        ].transform("sum")
                    )

                    the_stat_bucket_weight[normalized_stat_bucket_weight_col] = (
                        the_stat_bucket_weight[stat_bucket_weight_col]
                        / the_stat_bucket_weight[sum_of_stat_bucket_weight_col]
                    )

                    # join with data
                    the_data_at_relevant_grain = the_data_at_relevant_grain.merge(
                        the_stat_bucket_weight,
                        on=relevant_rank_column,
                        how="inner",
                    )
                    relevant_forecast_data = forecast_data.merge(
                        the_group[dimensions], on=dimensions, how="inner"
                    )
                    if relevant_forecast_data.empty:
                        logger.warning("Forecast data not available for the intersection")
                        continue

                    # left join with forecast data - forecast will be available for only selected partial weeks
                    the_data_at_relevant_grain = the_data_at_relevant_grain.merge(
                        relevant_forecast_data,
                        on=[version_col] + dimensions + [partial_week_col],
                        how="left",
                    )

                    # ==== IS RAMP UP PERIOD ==== #
                    # get ramp up period data
                    ramp_up_period_df = the_data_at_relevant_grain[
                        the_data_at_relevant_grain[is_ramp_up_period_col] == 1
                    ]

                    # calculate npi profile l0
                    ramp_up_period_df[ramp_up_profile_disagg_to_PW_col] = (
                        ramp_up_period_df[default_profile_col]
                        * ramp_up_period_df[normalized_stat_bucket_weight_col]
                    )

                    # normalize this for the ramp up period
                    ramp_up_period_df[normalized_ramp_up_profile_disagg_to_PW_col] = (
                        ramp_up_period_df[ramp_up_profile_disagg_to_PW_col]
                        / ramp_up_period_df[ramp_up_profile_disagg_to_PW_col].sum()
                    )

                    # disaggregate user defined ramp up volume
                    ramp_up_period_df[disagg_ramp_up_volume_col] = (
                        ramp_up_period_df[ramp_up_volume_col]
                        * ramp_up_period_df[normalized_ramp_up_profile_disagg_to_PW_col]
                    )

                    # disaggregate initial build
                    filter_clause = ramp_up_period_df[lifecycle_bucket_col] == 1
                    ramp_up_period_df[disagg_initial_build_col] = np.where(
                        filter_clause,
                        ramp_up_period_df[initial_build_col]
                        * ramp_up_period_df[normalized_stat_bucket_weight_col],
                        0,
                    )

                    # calculate npi forecast
                    ramp_up_period_df[output_measure] = (
                        ramp_up_period_df[disagg_ramp_up_volume_col]
                        + ramp_up_period_df[disagg_initial_build_col]
                    )

                    # ==== BEYOND RAMP UP PERIOD ==== #
                    # get beyond ramp up period data
                    beyond_ramp_up_period_df = the_data_at_relevant_grain[
                        the_data_at_relevant_grain[is_ramp_up_period_col] == 0
                    ]

                    # multiply like item forecast with scaling factor to get output
                    beyond_ramp_up_period_df[output_measure] = np.where(
                        beyond_ramp_up_period_df[like_item_forecast_col].notna(),
                        beyond_ramp_up_period_df[scaling_factor_col]
                        * beyond_ramp_up_period_df[like_item_forecast_col],
                        np.nan,
                    )

                    # combine both outputs and append to master list
                    the_data_at_relevant_grain = pd.concat(
                        [
                            ramp_up_period_df,
                            beyond_ramp_up_period_df,
                        ],
                        ignore_index=True,
                    )

                    # select the relevant columns
                    the_result = the_data_at_relevant_grain[
                        cols_required_in_output
                    ].drop_duplicates()

                # file_name = "_".join(_)
                # the_data_at_relevant_grain.to_csv(
                #     file_name + ".csv", index=False
                # )
                master_data_list.append(the_result)
            except Exception as e:
                logger.exception(e)

        output_df = concat_to_dataframe(master_data_list)

        if len(output_df) == 0:
            logger.warning("output_df is empty....")
            NPIForecast = pd.DataFrame(columns=cols_required_in_output)
        else:
            output_df[version_col] = input_version

            NPIForecast = output_df[cols_required_in_output]

            # Drop rows where all specified columns are null
            NPIForecast.dropna(subset=all_output_measures, how="all", inplace=True)

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        NPIForecast = pd.DataFrame(columns=cols_required_in_output)

    return NPIForecast
