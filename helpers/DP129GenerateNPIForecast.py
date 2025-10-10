"""Generate NPI Forecast for Flexible NPI.

Pseudocode:
    Version: 0.0.0
    --------------------
        - Case 1.1:
            - If NPI Forecast Generation Method L0 is Like Item Based and PLC Profile not assigned.
            - PLC Profile: Copy 'Ramp up profile Final L0' ('LaunchProfile' input) to 'NPI Profile L0' for 'NPI Ramp Up Period L0'.
            - 'NPI Ramp Up Bucket L0' could be month/planning month/week. Follow the bucket with same mentioned bucket.
            - Calculate 'NPI Profile Normalized L0' which is Normalize the copied profile (x/sum(all x)).
            - Calculate 'NPI Fcst L0' which is 'Normalized'*'User Defined Ramp Up Volume L0'*+Initial build.
            - Initial build is 'User Defined Ramp Up Volume L0' if not present then 'System Defined Ramp Up Volume L0'.
            - Dissagregate the Calculated fcst to partial week level.

            - After the 'NPI Ramp Up Period L0', copy 'Like Item Fcst L0'.
            - Disaggregate 'Like Item Fcst L0' to partial week level.
            - Multiple Disaggregated 'Like Item Fcst L0' with the scaling factor.
        - Case 1.2:
            - If NPI Forecast Generation Method L0 is Like Item Based and PLC Profile is assigned
            - Only change the PLC Profile. Take the PLC profile from 'PLCProfile' input based on user selected profile.
            - Follow the case 1.1
        Case 2:
            - If NPI Forecast Generation Method L0 is Manual
            - Do not copy like item fcst to NPI Fcst.
            - Follow the case 1.1


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

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")


def to_datetime_safely(df, col, format=None):
    """Convert the date time column safely."""
    if format is None:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception as e:
            logger.error(f"Unsupported date format for column: {col}. Please specify the 'format'.")
            raise ValueError(f"Invalid {col} format for input columns: {df.columns}. Error: {e}")
    else:
        try:
            df[col] = pd.to_datetime(df[col], format=format)
        except ValueError as e:
            try:
                logger.warning(f"{col} conversion has failed using format: {format}. Error: {e}...")
                logger.warning(f"Reattempting the {col} conversion with default date format...")
                df[col] = pd.to_datetime(df[col])
                logger.info(f"Successfully converted the {col} format.")
            except ValueError as ex:
                logger.error(f"Cannot convert the {col}. Error: {ex}")
                raise ValueError("Invalid {col} format for input columns: {df.columns}.")
    return df


def get_relevant_ramp_up_profile(
    Data,
    LaunchProfile,
    ramp_up_profile_final_col,
    PLCProfile,
    plc_profile_value,
    plc_profile_col,
    ramp_up_bucket,
    def_profile_col,
    plc_time_bucket_col,
    lifecycle_bucket_col,
):
    """Get relevant ramp up profile."""
    Profile = pd.DataFrame()

    if (
        pd.isna(plc_profile_value)
        or pd.isnull(plc_profile_value)
        or (plc_profile_value in [np.nan, "", " ", "NA", "na", None])
        or str(plc_profile_value).strip() == ""
    ):
        # If no profile mentioned then LaunchProfile
        common_cols = list(set(LaunchProfile.columns) & set(Data.columns))
        Profile = pd.merge(
            LaunchProfile, Data[common_cols].drop_duplicates(), on=common_cols, how="inner"
        )
        Profile.rename(columns={ramp_up_profile_final_col: def_profile_col}, inplace=True)

    else:
        # If profile mentioned then filter accordingly
        Profile = PLCProfile[
            (PLCProfile[plc_profile_col] == plc_profile_value)
            & (PLCProfile[plc_time_bucket_col] == ramp_up_bucket)
        ]

    Profile[[lifecycle_bucket_col, def_profile_col]].drop_duplicates().reset_index(drop=True)

    # Formatting the bucket
    Profile[lifecycle_bucket_col] = Profile[lifecycle_bucket_col].str.extract("(\d+)").astype(int)

    return Profile[[lifecycle_bucket_col, def_profile_col]].drop_duplicates().reset_index(drop=True)


col_mapping = {
    "NPI Level Profile L0": float,
    "NPI Level Profile Normalized L0": float,
    "NPI Fcst L0": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    # Paramas
    NPIStartDateFormat=None,
    NPIEndDateFormat=None,
    TimeMasterKeyFormat=None,
    # Data
    SelectedCombinations: pd.DataFrame = None,
    Parameters: pd.DataFrame = None,
    LaunchProfile: pd.DataFrame = None,
    PLCProfile: pd.DataFrame = None,
    LikeItemFcst: pd.DataFrame = None,
    NumDays: pd.DataFrame = None,
    TimeMaster: pd.DataFrame = None,
    df_keys=None,
):
    """Entry point of the script."""
    plugin_name = "DP129GenerateNPIForecast"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    initiative_col = "Initiative.[Initiative]"
    data_object_col = "Data Object.[Data Object]"
    daykey_col = "Time.[DayKey]"
    week_col = "Time.[Week]"
    partial_week_col = "Time.[Partial Week]"
    partialkey_week_col = "Time.[PartialWeekKey]"
    month_col = "Time.[Month]"
    pl_month_col = "Time.[Planning Month]"
    plc_time_bucket_col = "PLC Profile.[PLC Time Bucket]"
    lifecycle_bucket_col = "Lifecycle Time.[Lifecycle Bucket]"

    npi_item_col = "Item.[NPI Item]"
    npi_account_col = "Account.[NPI Account]"
    npi_channel_col = "Channel.[NPI Channel]"
    npi_region_col = "Region.[NPI Region]"
    npi_pnl_col = "PnL.[NPI PnL]"
    npi_location_col = "Location.[NPI Location]"
    npi_demand_domain_col = "Demand Domain.[NPI Demand Domain]"

    npi_association_col = "NPI Association L0"
    npi_bucket_col = "NPI Bucket L0"
    start_date_col = "Start Date L0"
    end_date_col = "End Date L0"
    npi_fcst_gen_method_col = "NPI Forecast Generation Method L0"
    user_def_npi_profile_col = "User Defined NPI Profile L0"
    plc_profile_col = "PLC Profile.[PLC Profile]"
    npi_ramp_up_period_col = "NPI Ramp Up Period L0"
    initial_build_col = "Initial Build L0"
    disagg_initial_build_col = "Disaggregated Initial Build L0"
    sys_def_ramp_up_vol_col = "System Suggested Ramp Up Volume L0"
    user_def_ramp_up_vol_col = "User Defined Ramp Up Volume L0"
    scaling_factor_col = "Scaling Factor L0"
    npi_ramp_up_bucket_col = "NPI Ramp Up Bucket L0"
    user_def_npi_period_col = "User Defined NPI Period L0"
    user_def_tot_vol_col = "User Defined Total Volume L0"
    ramp_up_profile_final_col = "Ramp Up Profile Final L0"
    def_profile_col = "Default Profile"
    norm_profile_col = "Normalized Default Profile"
    disagg_def_profile_col = "Disaggregated Default Profile"
    disagg_norm_def_profile_col = "Disaggregated Normalized Default Profile"
    num_days_col = "Num Days"
    sum_of_stat_bucket_weight = "sum of stat bucket weight"
    norm_stat_bucket_weight = "normalized stat bucket weight"
    npi_profile_col = "NPI Level Profile L0"
    npi_profile_norm_col = "NPI Level Profile Normalized L0"
    npi_fcst_col = "NPI Fcst L0"
    like_item_fcst_col = "Like Item Fcst L0"
    gen_sys_npi_fcst_ass_col = "Generate System NPI Fcst Assortment L0"

    # output columns
    cols_required_in_output = [
        version_col,
        initiative_col,
        data_object_col,
        npi_item_col,
        npi_account_col,
        npi_channel_col,
        npi_region_col,
        npi_pnl_col,
        npi_location_col,
        npi_demand_domain_col,
        partial_week_col,
        npi_fcst_col,
        npi_profile_col,
        npi_profile_norm_col,
    ]

    # Empty output dataframe
    NPIFcst_output = pd.DataFrame(columns=cols_required_in_output)

    try:

        NPIStartDateFormat = (
            eval(NPIStartDateFormat) if NPIStartDateFormat == "None" else NPIStartDateFormat
        )
        NPIEndDateFormat = (
            eval(NPIEndDateFormat) if NPIEndDateFormat == "None" else NPIEndDateFormat
        )
        TimeMasterKeyFormat = (
            eval(TimeMasterKeyFormat) if TimeMasterKeyFormat == "None" else TimeMasterKeyFormat
        )

        logger.info(f"Data processing for slice {df_keys}")
        SelectedCombinations[gen_sys_npi_fcst_ass_col] = pd.to_numeric(
            SelectedCombinations[gen_sys_npi_fcst_ass_col], downcast="integer", errors="coerce"
        )
        SelectedCombinations = SelectedCombinations[
            SelectedCombinations[gen_sys_npi_fcst_ass_col] >= 1
        ]

        if len(SelectedCombinations) == 0:
            raise ValueError(
                f"SelectedCombinations input does not contains the valid data. Please validate the data for slice {df_keys}..."
            )

        # Filter selected intersection
        common_cols = list(set(SelectedCombinations.columns) & set(Parameters.columns))
        Parameters = pd.merge(
            Parameters,
            SelectedCombinations[common_cols].drop_duplicates(),
            on=common_cols,
            how="inner",
        )

        Parameters[npi_association_col] = pd.to_numeric(
            Parameters[npi_association_col], downcast="integer", errors="coerce"
        )

        # Associated intersections
        Parameters = Parameters[Parameters[npi_association_col] >= 1]

        # Following df Cant be empty
        if len(Parameters) == 0:
            raise ValueError(f"Parameters cannot be empty for slice {df_keys} ...")

        logger.debug(f"Total Parameters rows: {len(Parameters)} for slice {df_keys}")

        if len(NumDays) == 0:
            raise ValueError(f"NumDays cannot be empty for slice {df_keys} ...")

        # Date format conversion | If col is already in date format then it will be ignored.
        Parameters = to_datetime_safely(Parameters, start_date_col, NPIStartDateFormat)
        Parameters = to_datetime_safely(Parameters, end_date_col, NPIEndDateFormat)

        # Date conversion
        TimeMaster = to_datetime_safely(TimeMaster, daykey_col, format=TimeMasterKeyFormat)
        TimeMaster = to_datetime_safely(TimeMaster, partialkey_week_col, format=TimeMasterKeyFormat)

        # Start date and End date should be within supported period
        min_possible_date = pd.to_datetime(TimeMaster[daykey_col]).min()
        max_possible_date = pd.to_datetime(TimeMaster[daykey_col]).max()

        # Override user provided values - we cannot populate any data beyond the min/max value in time dimension | If start date is empty do not gen the fcst
        Parameters[end_date_col] = Parameters[end_date_col].fillna(max_possible_date)
        Parameters[start_date_col] = Parameters[start_date_col].clip(
            lower=min_possible_date, upper=max_possible_date
        )
        Parameters[end_date_col] = Parameters[end_date_col].clip(
            lower=min_possible_date, upper=max_possible_date
        )

        # Handle missing values
        Parameters.fillna(np.nan, inplace=True)
        Parameters.replace({"": np.nan, "  ": np.nan}, inplace=True)

        # Fill missing values with default values
        Parameters[initial_build_col] = Parameters[initial_build_col].fillna(0)
        Parameters[scaling_factor_col] = Parameters[scaling_factor_col].fillna(1)
        Parameters[npi_ramp_up_period_col] = Parameters[npi_ramp_up_period_col].fillna(0)
        Parameters[user_def_npi_period_col] = Parameters[user_def_npi_period_col].fillna(0)
        Parameters[sys_def_ramp_up_vol_col] = Parameters[sys_def_ramp_up_vol_col].fillna(0)
        Parameters[user_def_tot_vol_col] = Parameters[user_def_tot_vol_col].fillna(0)

        # Considering user defined ramp up vol over system defined ramp up vol
        Parameters[sys_def_ramp_up_vol_col] = Parameters[user_def_ramp_up_vol_col].where(
            Parameters[user_def_ramp_up_vol_col].notna(), Parameters[sys_def_ramp_up_vol_col]
        )

        # Dropping version col
        NumDays.drop(columns=[version_col], axis=1, inplace=True)
        LikeItemFcst.drop(columns=[version_col], axis=1, inplace=True)

        # Handle null fcst values
        LikeItemFcst[like_item_fcst_col] = pd.to_numeric(
            LikeItemFcst[like_item_fcst_col], downcast="float", errors="coerce"
        )
        LikeItemFcst[like_item_fcst_col] = LikeItemFcst[like_item_fcst_col].fillna(0)

        NumDays[num_days_col] = pd.to_numeric(NumDays[num_days_col], downcast="integer")

        if (len(NumDays.drop_duplicates()) == 0) or (NumDays[num_days_col].sum() == 0):
            raise ValueError("NumDays input is not valid, please check the data...")

        # --- Calculating NPI Fcst for each assortments
        NPIFcst_list = []
        for _, parameter_data in Parameters.reset_index(drop=True).iterrows():
            logger.info(f"Calculating NPI Fcst for slice {df_keys}")
            logger.debug(f"Calculating NPI Fcst for {parameter_data} for slice {df_keys}")
            # Values
            npi_fcst_gen_method = parameter_data[npi_fcst_gen_method_col]
            plc_profile_value = parameter_data[user_def_npi_profile_col]
            start_date = pd.to_datetime(parameter_data[start_date_col])
            end_date = pd.to_datetime(parameter_data[end_date_col])

            if pd.isna(start_date) or start_date in ["", np.nan, " "]:
                logger.warning(
                    f"{start_date_col} is missing. Can't generate the NPI Fcst for slice: {df_keys} for intersection: \n{parameter_data}"
                )
                continue

            if pd.isna(end_date):
                logger.warning(
                    f"{end_date_col} is missing. Considering full scope for slice: {df_keys} for intersection: \n{parameter_data}"
                )
                end_date = max_possible_date

            # Handle 'Like Item Based' fcst
            the_LikeItemFcst = pd.DataFrame()
            if npi_fcst_gen_method == "Like Item Based":
                # If user selected Like item based fcst, like item fcst is mandatory
                if len(LikeItemFcst) == 0:
                    logger.warning(f"Like Item Fcst is not available at all for slice {df_keys}...")
                    logger.warning(
                        f"Skipping the NPI Fcst calculation for the intersection for slice {df_keys}."
                    )
                    continue
                else:
                    # Filtering the relevant Like Item Fcst
                    common_cols = list(set(LikeItemFcst) & set(parameter_data.index))
                    the_LikeItemFcst = pd.merge(
                        LikeItemFcst,
                        pd.DataFrame(parameter_data).T[
                            common_cols
                        ],  # Creating dataframe->rotate to bring columns on head -> select relevant cols
                        on=common_cols,
                        how="inner",
                    )
                    if len(the_LikeItemFcst) == 0:
                        logger.warning(
                            f"Like Item Fcst is not available for slice {df_keys} for intersection: \n{parameter_data}..."
                        )
                        logger.warning("Skipping the NPI Fcst calculation for the intersection.")
                        continue
                # Values
                ramp_up_bucket = parameter_data[npi_ramp_up_bucket_col]

                if pd.isna(ramp_up_bucket) or ramp_up_bucket in [np.nan, "nan", None, ""]:
                    logger.warning(
                        f"{npi_ramp_up_bucket_col} is empty. Considering Planning Month as default for slice: {df_keys} for intersection: \n{parameter_data}"
                    )
                    ramp_up_bucket = "Partial Week"

                ramp_up_time_col = "Time.[" + str(ramp_up_bucket.strip()) + "]"
                ramp_up_period = parameter_data[npi_ramp_up_period_col]

                if pd.isna(ramp_up_period) or int(ramp_up_period) < 0:
                    logger.warning(
                        f"{npi_ramp_up_period_col} is empty. Considering '0' as default for slice: {df_keys} for intersection: \n{parameter_data}"
                    )
                    ramp_up_period = 0
                else:
                    ramp_up_period = int(ramp_up_period)

            elif npi_fcst_gen_method == "Manual":
                # Values
                ramp_up_bucket = parameter_data[npi_bucket_col]

                if pd.isna(ramp_up_bucket) or ramp_up_bucket in [np.nan, "nan", None, ""]:
                    logger.warning(
                        f"{npi_bucket_col} is empty. Considering Planning Month as default for slice: {df_keys} for intersection: \n{parameter_data}"
                    )
                    ramp_up_bucket = "Planning Month"

                ramp_up_time_col = "Time.[" + str(ramp_up_bucket.strip()) + "]"
                ramp_up_period = int(parameter_data[user_def_npi_period_col])

                if ramp_up_period == 0 or pd.isna(ramp_up_period):
                    logger.warning(
                        f"{npi_ramp_up_period_col} is empty. Considering '0' as default for slice: {df_keys} for intersection: \n{parameter_data}"
                    )
                    ramp_up_period = 0

                parameter_data[sys_def_ramp_up_vol_col] = parameter_data[user_def_tot_vol_col]

                if parameter_data[sys_def_ramp_up_vol_col] == 0:
                    logger.warning(
                        f"{user_def_tot_vol_col} is missing for slice {df_keys} for intersection: \n{parameter_data}"
                    )
                    logger.warning("Skipping the intersection...")
                    continue

            else:
                allowed_npi_fcst_gen_method = ["Manual", "Like Item Based"]
                logger.warning(
                    f"Invalid {npi_fcst_gen_method_col} value passed. Value: {npi_fcst_gen_method}. Allowed: {allowed_npi_fcst_gen_method} for slice {df_keys}"
                )
                logger.warning("Skipping the intersection...")
                continue

            # Relavant npi bucket data
            partial_week_master_data = TimeMaster[
                [partial_week_col, partialkey_week_col, week_col, month_col, pl_month_col]
            ].drop_duplicates()

            # Filter for the user defined period
            filter_clause = (partial_week_master_data[partialkey_week_col] >= start_date) & (
                partial_week_master_data[partialkey_week_col] <= end_date
            )

            partial_week_master_data = partial_week_master_data[filter_clause]

            # Open time series intersection for the NPI
            the_data = create_cartesian_product(
                pd.DataFrame(parameter_data).T,  # parameter_data is a series
                partial_week_master_data,
            )

            # ===---------------------------------- NPI Fcst for the Ramp Up Period
            # Get relevant profile to be applied on the NPI
            user_profile = get_relevant_ramp_up_profile(
                the_data,
                LaunchProfile,
                ramp_up_profile_final_col,
                PLCProfile,
                plc_profile_value,
                plc_profile_col,
                ramp_up_bucket,
                def_profile_col,
                plc_time_bucket_col,
                lifecycle_bucket_col,
            )

            # user_profile is must for manual npi
            if len(user_profile) == 0:
                if npi_fcst_gen_method == "Manual":
                    logger.warning(
                        f"Not a valid profile selected. Please select a valid profile to proceed for slice {df_keys} for intersection: \n{the_data.iloc[:1,:11]}"
                    )
                    logger.warning("Skipping the intersection...")
                    continue
                else:
                    logger.warning(
                        f"Profile data not found for slice {df_keys} for intersection: \n{parameter_data}..."
                    )
                    logger.warning("NPI Fcst calculation will be skipped for the Ramp Up period...")

            # Manual NPI needs to calculate for the ramp up period only | Applying ramp up period
            user_profile = user_profile[user_profile[lifecycle_bucket_col] <= ramp_up_period]

            # Normalize the profile
            user_profile[norm_profile_col] = user_profile[def_profile_col] / np.sum(
                user_profile[def_profile_col]
            )
            # Note: Direct normalizing at one shot after disaggregation will not give a correct result.

            # Creating life cycle bucket col for merge
            the_data.sort_values(by=partialkey_week_col, ascending=True, inplace=True)
            the_data[lifecycle_bucket_col] = pd.factorize(the_data[ramp_up_time_col])[0] + 1

            # Rest of the period will be for beyond ramp up period calculations
            the_data = the_data.merge(user_profile, on=[lifecycle_bucket_col], how="left")

            # Get partial week weightage
            the_data = pd.merge(the_data, NumDays, on=[partial_week_col], how="left")
            the_data[num_days_col] = the_data[num_days_col].fillna(0)

            # Creating weightage % | Normalizing the num days by ramp up bucket
            the_data[sum_of_stat_bucket_weight] = the_data.groupby(ramp_up_time_col)[
                num_days_col
            ].transform("sum")
            the_data[norm_stat_bucket_weight] = (
                the_data[num_days_col] / the_data[sum_of_stat_bucket_weight]
            )

            # Disaggregate the default profile
            the_data[disagg_def_profile_col] = (
                the_data[def_profile_col] * the_data[norm_stat_bucket_weight]
            )

            # Disaggregate the normalized profile
            the_data[disagg_norm_def_profile_col] = (
                the_data[norm_profile_col] * the_data[norm_stat_bucket_weight]
            )

            # Disaggregate the Ramp up volume | This is the NPI Fcst without initial build for the ramp up period
            the_data[npi_fcst_col] = (
                the_data[sys_def_ramp_up_vol_col] * the_data[disagg_norm_def_profile_col]
            )

            # Disaggregate initial build
            the_data[disagg_initial_build_col] = (
                the_data[initial_build_col] * the_data[norm_stat_bucket_weight]
            ).where(the_data[lifecycle_bucket_col] == 1, 0)

            # Output NPI Fcst | Add initial build
            the_data[npi_fcst_col] = the_data[npi_fcst_col] + the_data[disagg_initial_build_col]

            # ===----------------------------- Beyond the ramp up period
            # Applicable only if NPI Generation method is 'Like Item Based'
            common_cols = list(set(the_data.columns) & set(LikeItemFcst.columns))
            the_data = pd.merge(the_data, LikeItemFcst, on=common_cols, how="left")

            the_data[npi_fcst_col] = (
                the_data[like_item_fcst_col] * the_data[scaling_factor_col]
            ).where(
                (the_data[lifecycle_bucket_col] > ramp_up_period)
                & (npi_fcst_gen_method == "Like Item Based"),
                the_data[npi_fcst_col],
            )

            # Restricting the open intersection within the like item fcst availability or Manual fcst, which ever is available
            the_data.dropna(subset=[npi_fcst_col], inplace=True)

            # Output for NPI Profile L0
            the_data[npi_profile_col] = the_data[disagg_def_profile_col]

            # Output for NPI Profile Normalized L0
            the_data[npi_profile_norm_col] = the_data[disagg_norm_def_profile_col]

            # Required columns
            the_NPIFcst = the_data[cols_required_in_output]

            NPIFcst_list.append(the_NPIFcst)

        NPIFcst_output = concat_to_dataframe(NPIFcst_list)
        logger.info("NPI Fcst calculation completed.")

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        logger.exception(e)
        return NPIFcst_output

    return NPIFcst_output
