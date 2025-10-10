"""Generate Cannibalization Impact Plugin for Flexible NPI.

Pseudocode:
    --------------------
    - Calculating the Cannibalization Impact

    - Calculating Cannibalization Independence date
    - Iterating for all the rows of selectedInitiative:
    - Calculating the split % for all the selceted combination of the initiative level

    - Assigning the Concated dfs to the final output
"""

import datetime
import logging
from functools import reduce

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta  # type: ignore
from o9Reference.common_utils.decorators import map_output_columns_to_dtypes
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")

col_mapping = {
    "640 Initiative Cannibalization Impact.[Cannib Impact L0]": float,
    "640 Initiative Cannibalization Impact.[Cannib Profile L0]": float,
    "Cannibalization Independence Date L0": "datetime64[ns]",
    "System Cannib Split %": float,
}


def calculate_cannibalization_independence_date(
    intro_date,
    cannibalization_period,
    bucket,
    time_dimension,
    PlanningMonthFormat,
):
    """Calculate the Cannibalization Independence Date based on the intro date, user-defined cannibalization period, and the profile bucket.

    Parameters:
    intro_date (pd.Timestamp): The introduction date of the item.
    cannibalization_period (int): The user-defined period for cannibalization (in months or weeks).
    bucket (str): The cannibalization profile bucket type ('Planning Month', 'Month', 'Week').
    time_dimension (pd.DataFrame): The DataFrame containing time-related information.

    Returns:
    pd.Timestamp: The last day of the month or week following the defined period from the intro date.
    """
    if pd.isnull(intro_date) or pd.isnull(cannibalization_period):
        return None  # Return None if intro date or period is invalid

    cannibalization_period -= 1

    if bucket == "Week":
        # Step 1: Find the week containing the intro_date
        intro_week = time_dimension[
            time_dimension["Time.[Day]"] == intro_date.strftime("%Y-%m-%d")
        ]["Time.[Week]"]

        if intro_week.empty:
            raise ValueError("Intro Date not found in TimeDimension DataFrame.")

        intro_week = intro_week.iloc[0]

        # Step 2: Calculate the future week by adding the cannibalization period in weeks
        future_week_start = time_dimension[time_dimension["Time.[Week]"] == intro_week][
            "Time.[Day]"
        ].min()
        future_week_start_date = pd.to_datetime(future_week_start) + relativedelta(
            weeks=cannibalization_period
        )
        future_week_start_str = future_week_start_date.strftime("%Y-%m-%d")

        # Step 3: Find the future week in the TimeDimension DataFrame
        future_week = time_dimension[time_dimension["Time.[Day]"] == future_week_start_str][
            "Time.[Week]"
        ]

        if future_week.empty:
            raise ValueError("Future Week not found in TimeDimension DataFrame.")

        future_week = future_week.iloc[0]

        # Step 4: Retrieve the last day of the target week
        last_day = time_dimension[time_dimension["Time.[Week]"] == future_week]["Time.[Day]"].max()

        return pd.to_datetime(last_day) if pd.notnull(last_day) else None

    elif bucket == "Planning Month":

        # Find the Planning Month for the intro_date

        intro_month = time_dimension[
            time_dimension["Time.[Day]"] == intro_date.strftime("%Y-%m-%d")
        ]["Time.[Planning Month]"]

        if intro_month.empty:
            raise ValueError("Intro Date's Planning Month not found in TimeDimension DataFrame.")

        intro_month = intro_month.iloc[0]

        # Convert Planning Month to a Timestamp
        intro_month_start = convert_intro_month(intro_month, PlanningMonthFormat)

        # Calculate the future Planning Month
        future_month_date = intro_month_start + relativedelta(months=cannibalization_period)

        target_month = future_month_date.strftime("%b-%Y")

        # Retrieve the last day of the target Planning Month

        last_day = time_dimension[
            time_dimension["Time.[Planning Month]"].dt.strftime("%b-%Y") == target_month
        ]["Time.[Day]"].max()

        return pd.to_datetime(last_day) if pd.notnull(last_day) else None

    elif bucket == "Month":

        # Calculate the future date based on the bucket type

        future_date = intro_date + relativedelta(months=cannibalization_period)

        target_month = future_date.strftime("%b-%Y")

        # Retrieve the last day of the target Month

        last_day = time_dimension[
            time_dimension["Time.[Month]"].dt.strftime("%b-%Y") == target_month
        ]["Time.[Day]"].max()

        return pd.to_datetime(last_day) if pd.notnull(last_day) else None

    else:

        raise ValueError("Unknown Cannibalization Profile Bucket")


def convert_intro_month(intro_month, PlanningMonthFormat="M%b-%Y"):
    """Execute multiple ways to convert intro_month to a datetime format.

    - First: Uses the user-specified `PlanningMonthFormat`.
    - Second: Falls back to the system default format ("%b-%Y").
    - Third: Uses `coerce` to avoid errors if the input is invalid.
    - Finally: Raises an error if all methods fail.
    """
    if isinstance(intro_month, pd.Timestamp):
        return intro_month

    try:
        # First attempt: Using user-specified format
        return pd.to_datetime(intro_month, format=PlanningMonthFormat)

    except ValueError:
        logger.info("Conversion failed with user-specified planning month date format.")

    try:
        # Second attempt: Using system default format
        logger.info("Converting using default system date format")
        return pd.to_datetime(f"01-{intro_month}", format="%b-%Y")

    except ValueError:
        logger.info("Conversion failed with system default planning month date format.")

    # Third attempt: Using `coerce` to avoid errors
    logger.info("Converting using coerce and default Python date format")
    intro_month_start = pd.to_datetime(intro_month, errors="coerce")

    if pd.isna(intro_month_start):
        raise ValueError(
            f"Failed to convert the date '{intro_month}'. Please modify the code/data."
        )

    return intro_month_start


def to_date_time(df, target_col, source_col, format="%Y-%m-%d"):
    """Convert date time optimally."""
    df[target_col] = df[source_col].astype("string[pyarrow]")
    df[target_col] = pd.to_datetime(df[target_col], format=format, cache=True)
    return df


def merge_two(df1, df2_key):
    """Merge two dfs."""
    key, df2 = df2_key
    df2 = df2.loc[:, ~df2.columns.duplicated()]
    # if user levels are same, that means, the req column is already present
    if len(df2.columns) == 1:
        return df1
    return pd.merge(df1, df2, on=key, how="left")


def process_assortment_npi(AssortmentFinal_req, key_level_cols, key_cols_to_retain):
    """Compute Split % for AssortmentFinal_req based on the Assortment Final column."""
    # Check if AssortmentFinal_req is empty
    if AssortmentFinal_req.empty:
        logger.warning("AssortmentFinal_req is empty. Returning an empty DataFrame.")
        return AssortmentFinal_req

    # Check if all key columns exist in AssortmentFinal_req
    missing_cols = [col for col in key_level_cols if col not in AssortmentFinal_req.columns]
    if missing_cols:
        logger.warning(
            f"Missing key columns in AssortmentFinal_req: {missing_cols}. Returning unmodified DataFrame."
        )
        return AssortmentFinal_req

    # Compute group-wise sum for Split % calculation
    AssortmentFinal_req["Group Sum"] = AssortmentFinal_req.groupby(key_level_cols)[
        "Assortment Final"
    ].transform("sum")

    # Compute Split %
    AssortmentFinal_req["Split %"] = (
        AssortmentFinal_req["Assortment Final"] / AssortmentFinal_req["Group Sum"]
    )

    # Drop rows where "Assortment Final" is missing
    AssortmentFinal_req = AssortmentFinal_req.dropna(subset=["Assortment Final"])

    # Identify columns to drop (all columns except key_cols_to_retain)
    cols_to_drop = [col for col in AssortmentFinal_req.columns if col not in key_cols_to_retain]

    # Drop the unnecessary columns
    AssortmentFinal_req = AssortmentFinal_req.drop(
        columns=[col for col in cols_to_drop if col in AssortmentFinal_req.columns]
    )

    return AssortmentFinal_req


def get_key_columns(cannibitem_req):
    """
    Extract and returns key column mappings dynamically from the given dataframe.

    Args:
        cannibitem_req (pd.DataFrame): DataFrame containing columns with names to be mapped.

    Returns:
        tuple: A tuple containing key column names in the correct order.
    """
    # List of columns that need to be mapped
    key_mappings = ["Item", "Account", "Channel", "Region", "PnL", "Demand Domain", "Location"]

    # Dictionary to store the mapped key columns
    key_cols = {}

    for col in cannibitem_req.columns:
        for key in key_mappings:
            if col.startswith(key):  # Check if the column starts with the key (e.g., "Item.")
                key_var = f"key_{key.replace(' ', '_')}_col"  # Format variable name
                key_cols[key_var] = col  # Assign the corresponding column name

    # Return values in the correct order
    return (
        key_cols.get("key_Item_col", None),
        key_cols.get("key_Account_col", None),
        key_cols.get("key_Channel_col", None),
        key_cols.get("key_Region_col", None),
        key_cols.get("key_PnL_col", None),
        key_cols.get("key_Demand_Domain_col", None),
        key_cols.get("key_Location_col", None),
    )


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    SelectedCombinations,
    CannibItem,
    StatFcst,
    NPIFcst,
    DefaultProfile,
    Parameters,
    TimeDim,
    RegionMaster,
    PnLMaster,
    DemandDomainMaster,
    ItemMaster,
    AccountMaster,
    ChannelMaster,
    LocationMaster,
    AssortmentFinal,
    InitiativeLevels,
    df_keys,
    PlanningMonthFormat=None,
):
    """DP133GenerateCannibalizationImpact plugin."""
    plugin_name = "DP133GenerateCannibalizationImpact"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    try:

        if (
            len(DefaultProfile) == 0
            or len(NPIFcst) == 0
            or len(Parameters) == 0
            or len(StatFcst) == 0
            or len(DefaultProfile) == 0
            or len(SelectedCombinations) == 0
        ):
            logger.warning("One of the inputs is Null. Exiting : {} ...".format(df_keys))
            logger.warning("Returning empty dataframe for this slice")
            cannibalizedForecast = pd.DataFrame()
            return cannibalizedForecast

        logger.info("Selecting required columns from input dataframes ...")

        # Configurables
        data_obj = "Data Object.[Data Object]"
        initiative_col = "Initiative.[Initiative]"
        item_col = "Item.[NPI Item]"
        location_col = "Location.[NPI Location]"
        account_col = "Account.[NPI Account]"
        region_col = "Region.[NPI Region]"
        demand_domain_col = "Demand Domain.[NPI Demand Domain]"
        pnl_col = "PnL.[NPI PnL]"
        channel_col = "Channel.[NPI Channel]"

        from_data_obj = "from.[Data Object].[Data Object]"
        from_initiative = "from.[Initiative].[Initiative]"
        from_item_col = "from.[Item].[NPI Item]"
        from_location_col = "from.[Location].[NPI Location]"
        from_account_col = "from.[Account].[NPI Account]"
        from_region_col = "from.[Region].[NPI Region]"
        from_pnl_col = "from.[PnL].[NPI PnL]"
        from_channel_col = "from.[Channel].[NPI Channel]"
        from_demand_domain = "from.[Demand Domain].[NPI Demand Domain]"
        to_item_col = "to.[Item].[NPI Item]"
        to_partial_week_col = "to.[Time].[Partial Week]"

        day_col = "Time.[Day]"
        daykey_col = "Time.[DayKey]"
        week_col = "Time.[Week]"
        weekkey_col = "Time.[WeekKey]"
        partial_week_col = "Time.[Partial Week]"
        month_col = "Time.[Month]"
        monthkey_col = "Time.[MonthKey]"
        planning_month_col = "Time.[Planning Month]"

        version_col = "Version.[Version Name]"
        assortment_col = "Generate Cannib Impact Assortment L0"
        cannib_profile = "640 Initiative Cannibalization Impact.[Cannib Profile L0]"
        cannib_impact = "640 Initiative Cannibalization Impact.[Cannib Impact L0]"
        intro_date = "Start Date L0"
        Cannibalization_Profile = "Cannibalization Profile L0"
        Cannibalization_Profile_Bucket = "Cannibalization Bucket L0"
        PLC_Profile = "PLC Profile.[PLC Profile]"
        PLC_Time_Bucket = "PLC Profile.[PLC Time Bucket]"
        user_def_cannib_period = "User Defined Cannibalization Period L0"
        Cannib_Independence_Date = "Cannibalization Independence Date"
        stat_fcst = "Stat Fcst NPI BB L0"
        default_profile = "Default Profile"
        npi_fcst = "NPI Fcst Final L0"
        from_demand_domain = "from.[Demand Domain].[NPI Demand Domain]"

        cannib_independence_dt_col = "Cannibalization Independence Date L0"
        cannib_Item_L0 = "635 Initiative Cannibalized Item Match.[Final Cannib Item L0]"
        split_col_name = "Cannib System Split %"
        assortmentfinal_col_name = "NPI Planning Assortment by Level"
        independence_dt_pllvl_output_name = "Cannibalization Independence Date Planning Level"
        assortment_final = "Assortment Final"
        assortmentfinal_col_name = "NPI Planning Assortment by Level"

        pl_item_col = "Item.[Planning Item]"
        pl_loc_col = "Location.[Planning Location]"
        planning_channel_col = "Channel.[Planning Channel]"
        planning_account_col = "Account.[Planning Account]"
        planning_pnl_col = "PnL.[Planning PnL]"
        planning_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
        planning_region_col = "Region.[Planning Region]"
        planning_location_col = "Location.[Planning Location]"

        all_splits = []
        # output measures
        cols_req_in_output = [
            version_col,
            from_data_obj,
            from_initiative,
            from_account_col,
            from_pnl_col,
            from_region_col,
            from_channel_col,
            from_demand_domain,
            from_location_col,
            to_partial_week_col,
            to_item_col,
            from_item_col,
            cannib_profile,
            cannib_impact,
        ]

        cols_req_in_output_independence = [
            version_col,
            data_obj,
            initiative_col,
            item_col,
            account_col,
            channel_col,
            location_col,
            region_col,
            pnl_col,
            demand_domain_col,
            cannib_independence_dt_col,
        ]

        cols_req_in_output_split = [
            version_col,
            data_obj,
            initiative_col,
            pl_item_col,
            planning_account_col,
            planning_channel_col,
            planning_demand_domain_col,
            planning_location_col,
            planning_pnl_col,
            planning_region_col,
            assortmentfinal_col_name,
            split_col_name,
            independence_dt_pllvl_output_name,
        ]

        cannibalizedForecast = pd.DataFrame(columns=cols_req_in_output)
        Cannibalization_Independence_Date = pd.DataFrame(columns=cols_req_in_output_independence)
        Splits = pd.DataFrame(columns=cols_req_in_output_split)

        SelectedCombinations = SelectedCombinations[SelectedCombinations[assortment_col] == 1]

        Parameters = pd.merge(
            SelectedCombinations,
            Parameters,
            how="inner",
            on=[
                version_col,
                data_obj,
                initiative_col,
                account_col,
                channel_col,
                region_col,
                pnl_col,
                demand_domain_col,
                location_col,
                item_col,
            ],
        )
        Parameters.drop(
            columns=assortment_col,
            axis=1,
            inplace=True,
        )

        Parameters[intro_date] = pd.to_datetime(Parameters[intro_date])

        TimeDim = to_date_time(TimeDim, day_col, daykey_col)

        # here the monthkey column is getting changed
        TimeDim = to_date_time(TimeDim, month_col, monthkey_col)

        DefaultProfile[default_profile] = DefaultProfile[default_profile].fillna(0)

        initiativelevels_filtered = pd.merge(
            InitiativeLevels,
            SelectedCombinations[
                ["Data Object.[Data Object]", "Initiative.[Initiative]"]
            ],  # selecting only relevant columns
            how="inner",  # 'inner' merge ensures only matching rows are kept
            left_on=[
                "Data Object.[Data Object]",
                "Initiative.[Initiative]",
            ],  # the columns from Initiativelevels
            right_on=[
                "Data Object.[Data Object]",
                "Initiative.[Initiative]",
            ],  # the columns from Selectedcombinations
        )

        initiativelevels_filtered = initiativelevels_filtered.drop_duplicates()

        # Check if the resulting DataFrame is empty
        if initiativelevels_filtered.empty:
            logger.warning("No matching rows found after merge. The resulting DataFrame is empty.")

        # independence date df
        Cannibalization_Independence_Date = Parameters.copy()

        # Calculating the CaanibImpact

        B00Params = pd.merge(
            Parameters,
            DefaultProfile,
            how="inner",
            left_on=[
                version_col,
                Cannibalization_Profile,
                Cannibalization_Profile_Bucket,
            ],
            right_on=[
                version_col,
                PLC_Profile,
                PLC_Time_Bucket,
            ],
        )
        B00Params.drop(
            [PLC_Profile, PLC_Time_Bucket],
            axis=1,
            inplace=True,
        )

        B00Params = B00Params.sort_values(
            by=[
                version_col,
                account_col,
                channel_col,
                region_col,
                pnl_col,
                demand_domain_col,
                location_col,
                item_col,
                intro_date,
                Cannibalization_Profile,
                Cannibalization_Profile_Bucket,
                user_def_cannib_period,
            ]
        ).reset_index(drop=True)
        B00Params["Offset"] = B00Params.groupby(
            [
                version_col,
                account_col,
                channel_col,
                region_col,
                pnl_col,
                demand_domain_col,
                location_col,
                item_col,
                intro_date,
                Cannibalization_Profile,
                Cannibalization_Profile_Bucket,
                user_def_cannib_period,
            ],
            observed=False,
        ).cumcount()

        # Divide based on the Cannibalization Profile Bucket
        Bucket_set = set(Parameters[Cannibalization_Profile_Bucket].tolist())
        Final_df = pd.DataFrame()

        # Ensure 'Intro Date' is datetime64[ns] in both DataFrames
        Parameters[intro_date] = pd.to_datetime(Parameters[intro_date])

        # Removing timestamp from intro date since it needs to be merged with TimeDimension
        B00Params[intro_date] = pd.to_datetime(B00Params[intro_date], format="%d-%b-%Y")
        B00Params[intro_date] = B00Params[intro_date].dt.normalize()

        # Define the columns for which we want unique intersections
        unique_cols = [
            version_col,
            data_obj,
            initiative_col,
            item_col,
            account_col,
            channel_col,
            region_col,
            pnl_col,
            location_col,
            demand_domain_col,
            intro_date,
        ]

        common_cols = [
            version_col,
            data_obj,
            initiative_col,
            item_col,
            account_col,
            channel_col,
            region_col,
            pnl_col,
            demand_domain_col,
            location_col,
            intro_date,
            user_def_cannib_period,
            Cannibalization_Profile,
            Cannibalization_Profile_Bucket,
        ]

        if PlanningMonthFormat in [
            None,
            "",
            " ",
            "NA",
            "NAN",
            "NONE",
            "none",
            "na",
        ]:
            logger.error(
                f"Invalid planning month format passed: {PlanningMonthFormat}. Please pass a correct Python format."
            )
            return pd.DataFrame(columns=cols_req_in_output)

        if "Planning Month" in Bucket_set:
            TimeDim = to_date_time(
                TimeDim,
                planning_month_col,
                planning_month_col,
                format=PlanningMonthFormat,
            )

            PlanningMonth_set = sorted(list(set(TimeDim[planning_month_col].tolist())))
            PMParams = B00Params[B00Params[Cannibalization_Profile_Bucket] == "Planning Month"]
            PMParams = pd.merge(
                PMParams,
                TimeDim[[day_col, planning_month_col]],
                left_on=intro_date,
                right_on=day_col,
                how="left",
            )

            # Custom function to replace offsets with planning months
            def replace_offsets_with_plans(group, plans):
                starting_plan = group[planning_month_col].iloc[0]
                start_index = plans.index(starting_plan)
                required_size = len(plans) - start_index
                group = group.head(required_size)
                plans_cycle = plans[start_index:] + plans[:start_index]
                group["Offset"] = [plans_cycle[i % len(plans_cycle)] for i in group["Offset"]]
                return group

            # Drop duplicates based on these columns
            unique_PMParams = PMParams.drop_duplicates(subset=unique_cols)

            unique_PMParams[Cannib_Independence_Date] = unique_PMParams.apply(
                lambda row: calculate_cannibalization_independence_date(
                    row[intro_date],
                    row[user_def_cannib_period],
                    row[Cannibalization_Profile_Bucket],
                    TimeDim,
                    PlanningMonthFormat,
                ),
                axis=1,
            )

            Cannibalization_Independence_Date["Start Date L0"] = pd.to_datetime(
                Cannibalization_Independence_Date["Start Date L0"]
            ).dt.normalize()

            unique_PMParams["Start Date L0"] = pd.to_datetime(
                unique_PMParams["Start Date L0"]
            ).dt.normalize()

            Cannibalization_Independence_Date = Cannibalization_Independence_Date.merge(
                unique_PMParams[common_cols + ["Cannibalization Independence Date"]],
                on=common_cols,
                how="left",
                suffixes=("", "_pm"),
            )

            # Apply the custom function to each group
            PMParams = (
                PMParams.groupby(
                    [
                        version_col,
                        account_col,
                        channel_col,
                        region_col,
                        pnl_col,
                        demand_domain_col,
                        location_col,
                        item_col,
                        intro_date,
                        planning_month_col,
                        Cannibalization_Profile,
                        Cannibalization_Profile_Bucket,
                        user_def_cannib_period,
                    ],
                    observed=False,
                )
                .apply(replace_offsets_with_plans, plans=PlanningMonth_set)
                .reset_index(drop=True)
            )
            PMParams.reset_index(drop=True, inplace=True)
            PMParams = PMParams.sort_values(
                by=[
                    version_col,
                    account_col,
                    channel_col,
                    region_col,
                    pnl_col,
                    demand_domain_col,
                    location_col,
                    item_col,
                    intro_date,
                    Cannibalization_Profile,
                    Cannibalization_Profile_Bucket,
                    user_def_cannib_period,
                    "Offset",
                ]
            )
            PMParams = pd.merge(
                PMParams,
                TimeDim[[partial_week_col, planning_month_col]].drop_duplicates(),
                how="inner",
                left_on=["Offset"],
                right_on=[planning_month_col],
            )
            PMParams.reset_index(drop=True, inplace=True)
            PMParams.drop(
                columns=[
                    user_def_cannib_period,
                    intro_date,
                    Cannibalization_Profile,
                    Cannibalization_Profile_Bucket,
                    day_col,
                    "Time.[Planning Month]_x",
                    "Time.[Planning Month]_y",
                    "Offset",
                ],
                axis=1,
                inplace=True,
            )
            Final_df = pd.concat([Final_df, PMParams], ignore_index=True)

        if "Month" in Bucket_set:
            Months_set = sorted(list(set(TimeDim[month_col].tolist())))
            MonthParams = B00Params[B00Params[Cannibalization_Profile_Bucket] == "Month"]
            MonthParams = pd.merge(
                MonthParams,
                TimeDim[[day_col, month_col]],
                left_on=intro_date,
                right_on=day_col,
                how="left",
            )

            # Custom function to replace offsets with planning months
            def replace_offsets_with_weeks(group, months):
                starting_plan = group[month_col].iloc[0]
                start_index = months.index(starting_plan)
                required_size = len(months) - start_index
                group = group.head(required_size)
                plans_cycle = months[start_index:] + months[:start_index]
                group["Offset"] = [plans_cycle[i % len(plans_cycle)] for i in group["Offset"]]
                return group

            # Drop duplicates based on these columns
            unique_MonthParams = MonthParams.drop_duplicates(subset=unique_cols)

            unique_MonthParams[Cannib_Independence_Date] = unique_MonthParams.apply(
                lambda row: calculate_cannibalization_independence_date(
                    row[intro_date],
                    row[user_def_cannib_period],
                    row[Cannibalization_Profile_Bucket],
                    TimeDim,
                    PlanningMonthFormat,
                ),
                axis=1,
            )

            Cannibalization_Independence_Date["Start Date L0"] = pd.to_datetime(
                Cannibalization_Independence_Date["Start Date L0"]
            ).dt.normalize()

            unique_MonthParams["Start Date L0"] = pd.to_datetime(
                unique_MonthParams["Start Date L0"]
            ).dt.normalize()

            Cannibalization_Independence_Date = Cannibalization_Independence_Date.merge(
                unique_MonthParams[common_cols + [Cannib_Independence_Date]],
                on=common_cols,
                how="left",
                suffixes=("", "_month"),
            )

            # Apply the custom function to each group
            MonthParams = MonthParams.groupby(
                [
                    version_col,
                    account_col,
                    channel_col,
                    region_col,
                    pnl_col,
                    demand_domain_col,
                    location_col,
                    item_col,
                    intro_date,
                    month_col,
                    Cannibalization_Profile,
                    Cannibalization_Profile_Bucket,
                    user_def_cannib_period,
                ]
            ).apply(replace_offsets_with_weeks, months=Months_set)
            MonthParams.reset_index(drop=True, inplace=True)
            MonthParams = MonthParams.sort_values(
                by=[
                    version_col,
                    account_col,
                    channel_col,
                    region_col,
                    pnl_col,
                    demand_domain_col,
                    location_col,
                    item_col,
                    intro_date,
                    Cannibalization_Profile,
                    Cannibalization_Profile_Bucket,
                    user_def_cannib_period,
                    month_col,
                ]
            )

            # Convert 'Time.[Month]' to datetime
            TimeDim = to_date_time(TimeDim, month_col, monthkey_col)

            # Merge MonthParams and TimeDimension on month to add partial weeks to MonthParams
            MonthParams = pd.merge(
                MonthParams,
                TimeDim[[partial_week_col, month_col]].drop_duplicates(),
                how="inner",
                left_on=["Offset"],
                right_on=[month_col],
            )
            MonthParams.reset_index(drop=True, inplace=True)
            MonthParams.drop(
                columns=[
                    user_def_cannib_period,
                    intro_date,
                    Cannibalization_Profile,
                    Cannibalization_Profile_Bucket,
                    "Time.[Month]_x",
                    "Time.[Month]_y",
                    "Offset",
                ],
                axis=1,
                inplace=True,
            )
            Final_df = pd.concat([Final_df, MonthParams], ignore_index=True)

        if "Week" in Bucket_set:
            TimeDim[week_col] = TimeDim[week_col].apply(
                lambda x: datetime.datetime.strptime(x, "%d-%b-%y")
            )
            Weeks_set = sorted(list(set(TimeDim[week_col].tolist())))
            WeekParams = B00Params[B00Params[Cannibalization_Profile_Bucket] == "Week"]
            WeekParams = pd.merge(
                WeekParams,
                TimeDim[[day_col, week_col]],
                left_on=intro_date,
                right_on=day_col,
                how="left",
            )

            # Custom function to replace offsets with planning months
            def replace_offsets_with_weeks(group, weeks):
                starting_plan = group[week_col].iloc[0]
                start_index = weeks.index(starting_plan)
                required_size = len(weeks) - start_index
                group = group.head(required_size)
                plans_cycle = weeks[start_index:] + weeks[:start_index]
                group["Offset"] = [plans_cycle[i % len(plans_cycle)] for i in group["Offset"]]
                return group

            # Drop duplicates based on these columns
            unique_WeekParams = WeekParams.drop_duplicates(subset=unique_cols)

            unique_WeekParams[Cannib_Independence_Date] = unique_WeekParams.apply(
                lambda row: calculate_cannibalization_independence_date(
                    row[intro_date],
                    row[user_def_cannib_period],
                    row[Cannibalization_Profile_Bucket],
                    TimeDim,
                    PlanningMonthFormat,
                ),
                axis=1,
            )

            Cannibalization_Independence_Date["Start Date L0"] = pd.to_datetime(
                Cannibalization_Independence_Date["Start Date L0"]
            ).dt.normalize()

            unique_WeekParams["Start Date L0"] = pd.to_datetime(
                unique_WeekParams["Start Date L0"]
            ).dt.normalize()

            Cannibalization_Independence_Date = Cannibalization_Independence_Date.merge(
                unique_WeekParams[common_cols + ["Cannibalization Independence Date"]],
                on=common_cols,
                how="left",
                suffixes=("", "_week"),
            )

            # Apply the custom function to each group
            WeekParams = WeekParams.groupby(
                [
                    version_col,
                    account_col,
                    channel_col,
                    region_col,
                    pnl_col,
                    demand_domain_col,
                    location_col,
                    item_col,
                    intro_date,
                    week_col,
                    Cannibalization_Profile,
                    Cannibalization_Profile_Bucket,
                    user_def_cannib_period,
                ]
            ).apply(replace_offsets_with_weeks, weeks=Weeks_set)
            WeekParams.reset_index(drop=True, inplace=True)
            WeekParams = WeekParams.sort_values(
                by=[
                    version_col,
                    account_col,
                    channel_col,
                    region_col,
                    pnl_col,
                    demand_domain_col,
                    location_col,
                    item_col,
                    intro_date,
                    Cannibalization_Profile,
                    Cannibalization_Profile_Bucket,
                    user_def_cannib_period,
                    "Offset",
                ]
            )

            # Convert 'Time.[Week]' to datetime64[ns]
            TimeDim = to_date_time(TimeDim, week_col, weekkey_col)

            WeekParams = pd.merge(
                WeekParams,
                TimeDim[[partial_week_col, week_col]].drop_duplicates(),
                how="inner",
                left_on=["Offset"],
                right_on=[week_col],
            )
            WeekParams.reset_index(drop=True, inplace=True)
            WeekParams.drop(
                columns=[
                    user_def_cannib_period,
                    intro_date,
                    Cannibalization_Profile,
                    Cannibalization_Profile_Bucket,
                    day_col,
                    "Time.[Week]_x",
                    "Time.[Week]_y",
                    "Offset",
                ],
                axis=1,
                inplace=True,
            )
            Final_df = pd.concat([Final_df, WeekParams], ignore_index=True)

        Final_df = pd.merge(
            NPIFcst,
            Final_df,
            how="inner",
            on=[
                version_col,
                account_col,
                channel_col,
                region_col,
                pnl_col,
                demand_domain_col,
                location_col,
                item_col,
                partial_week_col,
            ],
        )
        Final_df.rename(columns={"Default Profile": cannib_profile}, inplace=True)
        Final_df[cannib_impact] = Final_df[npi_fcst] * Final_df[cannib_profile]

        Final_df.drop(
            columns=[npi_fcst, "Initiative.[Initiative]_y", "Data Object.[Data Object]_y"],
            axis=1,
            inplace=True,
        )
        Final_df.rename(
            columns={
                "Data Object.[Data Object]_x": from_data_obj,
                "Initiative.[Initiative]_x": from_initiative,
                item_col: from_item_col,
                account_col: from_account_col,
                channel_col: from_channel_col,
                region_col: from_region_col,
                pnl_col: from_pnl_col,
                demand_domain_col: from_demand_domain,
                location_col: from_location_col,
                partial_week_col: to_partial_week_col,
            },
            inplace=True,
        )

        filtered_df = CannibItem.groupby(
            [
                version_col,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
                from_demand_domain,
                from_item_col,
            ],
            observed=False,
        ).filter(lambda x: x[to_item_col].nunique() >= 1)
        filtered_df = pd.merge(
            filtered_df,
            Final_df,
            how="inner",
            on=[
                version_col,
                from_data_obj,
                from_initiative,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
                from_demand_domain,
                from_item_col,
            ],
        )
        filtered_df = pd.merge(
            filtered_df,
            StatFcst,
            how="inner",
            left_on=[
                version_col,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
                from_demand_domain,
                to_item_col,
                to_partial_week_col,
            ],
            right_on=[
                version_col,
                account_col,
                channel_col,
                region_col,
                pnl_col,
                location_col,
                demand_domain_col,
                item_col,
                partial_week_col,
            ],
        )
        filtered_df.drop(
            columns=[
                account_col,
                channel_col,
                region_col,
                pnl_col,
                location_col,
                demand_domain_col,
                item_col,
                partial_week_col,
            ],
            axis=1,
            inplace=True,
        )
        filtered_df["sum"] = filtered_df.groupby(
            [
                version_col,
                from_data_obj,
                from_initiative,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
                from_item_col,
                from_demand_domain,
                to_partial_week_col,
                cannib_impact,
                cannib_profile,
            ],
            observed=False,
        )[stat_fcst].transform("sum")
        filtered_df["CImpact"] = (
            filtered_df[stat_fcst] * filtered_df[cannib_impact]
        ) / filtered_df["sum"]
        filtered_df.drop(
            columns=[
                "sum",
                stat_fcst,
                cannib_impact,
            ],
            axis=1,
            inplace=True,
        )
        filtered_df.rename(columns={"CImpact": cannib_impact}, inplace=True)

        Final_df = pd.merge(
            CannibItem,
            Final_df,
            how="inner",
            on=[
                version_col,
                from_data_obj,
                from_initiative,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
                from_demand_domain,
                from_item_col,
            ],
        )
        Final_df = Final_df.groupby(
            [
                version_col,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
                from_demand_domain,
                from_item_col,
            ],
            observed=False,
        ).filter(lambda x: x[to_item_col].nunique() == 1)
        cannibalizedForecast = pd.concat([Final_df, filtered_df], ignore_index=True)

        cannibalizedForecast.drop(
            columns=["Lifecycle Time.[Lifecycle Bucket]"], axis=1, inplace=True
        )

        cannibalizedForecast[cannib_profile] = cannibalizedForecast[cannib_profile].replace(
            0, np.nan
        )
        cannibalizedForecast[cannib_impact] = cannibalizedForecast[cannib_impact].replace(0, np.nan)
        cannibalizedForecast = cannibalizedForecast[cols_req_in_output]
        # Drop rows where both 'cannib_impact' and 'cannib_profile' are NaN
        cannibalizedForecast = cannibalizedForecast.dropna(
            subset=[cannib_impact, cannib_profile], how="all"
        )

        # fcst computation completed

        # Ensure the base column exists before filling
        if "Cannibalization Independence Date" not in Cannibalization_Independence_Date.columns:
            Cannibalization_Independence_Date["Cannibalization Independence Date"] = (
                pd.NaT
            )  # Initialize if missing

        # Fill missing values in sequence
        for col in [
            "Cannibalization Independence Date_month",
            "Cannibalization Independence Date_pm",
            "Cannibalization Independence Date_week",
        ]:
            if col in Cannibalization_Independence_Date.columns:
                Cannibalization_Independence_Date["Cannibalization Independence Date"] = (
                    Cannibalization_Independence_Date["Cannibalization Independence Date"].fillna(
                        Cannibalization_Independence_Date[col]
                    )
                )

        # Drop redundant columns
        Cannibalization_Independence_Date.drop(
            columns=[
                col
                for col in [
                    "Cannibalization Independence Date_month",
                    "Cannibalization Independence Date_pm",
                    "Cannibalization Independence Date_week",
                    intro_date,
                    user_def_cannib_period,
                ]
                if col in Cannibalization_Independence_Date.columns
            ],
            inplace=True,
        )
        Cannibalization_Independence_Date["Cannibalization Independence Date"] = pd.to_datetime(
            Cannibalization_Independence_Date["Cannibalization Independence Date"]
        )
        Cannibalization_Independence_Date.rename(
            columns={"Cannibalization Independence Date": cannib_independence_dt_col}, inplace=True
        )
        Cannibalization_Independence_Date = Cannibalization_Independence_Date[
            cols_req_in_output_independence
        ]
        # independence_date_df = Cannibalization_Independence_Date
        # filtering the cannib df
        # Step 1: Define column mapping for renaming
        column_mapping = {
            from_initiative: initiative_col,
            from_data_obj: data_obj,
            from_item_col: item_col,
            from_account_col: account_col,
            from_channel_col: channel_col,
            from_region_col: region_col,
            from_pnl_col: pnl_col,
            from_demand_domain: demand_domain_col,
            from_location_col: location_col,
        }

        # Rename columns in the copied DataFrame
        cannibItem_filtered = CannibItem.copy()
        cannibItem_filtered = cannibItem_filtered.rename(columns=column_mapping)
        merge_keys = list(column_mapping.values()) + [version_col]

        # Step 2: Perform filtering with parameters table
        # Keep only rows in cannibItem_filtered that are present in parameters
        cannibItem_filtered = cannibItem_filtered.merge(
            SelectedCombinations, on=merge_keys, how="inner"
        )
        # Dropping the specified columns
        cols_to_drop = [cannib_Item_L0, assortment_col]
        cannibItem_filtered = cannibItem_filtered.drop(
            columns=[col for col in cols_to_drop if col in cannibItem_filtered.columns]
        )

        # Define merge keys
        merge_keys = [
            version_col,
            data_obj,
            initiative_col,
            item_col,
            account_col,
            channel_col,
            region_col,
            pnl_col,
            demand_domain_col,
            location_col,
        ]

        # Ensure all keys exist in both DataFrames before merging
        missing_keys_cannib = [col for col in merge_keys if col not in cannibItem_filtered.columns]
        missing_keys_independence = [
            col for col in merge_keys if col not in Cannibalization_Independence_Date.columns
        ]

        if missing_keys_cannib or missing_keys_independence:
            logger.warning(f"Missing keys in CannibItem_renamed_2: {missing_keys_cannib}")
            logger.warning(f"Missing keys in independence_date_df: {missing_keys_independence}")
        else:
            # Merge the required columns into CannibItem_renamed_2
            cannibItem_filtered = cannibItem_filtered.merge(
                Cannibalization_Independence_Date[merge_keys + [cannib_independence_dt_col]],
                on=merge_keys,
                how="left",
            )

        (
            key_item_col,
            key_account_col,
            key_channel_col,
            key_region_col,
            key_pnl_col,
            key_dd_col,
            key_loc_col,
        ) = get_key_columns(cannibItem_filtered)

        rename_mapping = {
            "Item.[NPI Item]": "from.[Item].[NPI Item]",
            "to.[Item].[NPI Item]": "Item.[NPI Item]",
        }

        # Apply renaming manually
        cannibItem_filtered = cannibItem_filtered.rename(columns=rename_mapping)

        # List of NPI Level columns from InitiativeLevels that need to be mapped to cannibItem columns
        npi_level_columns = [
            "NPI Account Level",
            "NPI Channel Level",
            "NPI Region Level",
            "NPI PnL Level",
            "NPI Demand Domain Level",
            "NPI Location Level",
            "NPI Item Level",
        ]

        # List of corresponding cannibItem columns
        cannib_columns = [
            "Account.[NPI Account]",
            "Channel.[NPI Channel]",
            "Region.[NPI Region]",
            "PnL.[NPI PnL]",
            "Demand Domain.[NPI Demand Domain]",
            "Location.[NPI Location]",
            "Item.[NPI Item]",
        ]
        key_cols_to_retain = [
            "Version.[Version Name]",
            "Data Object.[Data Object]",
            "Initiative.[Initiative]",
            "Item.[Planning Item]",
            "Channel.[Planning Channel]",
            "Account.[Planning Account]",
            "PnL.[Planning PnL]",
            "Demand Domain.[Planning Demand Domain]",
            "Region.[Planning Region]",
            "Location.[Planning Location]",
            "Assortment Final",
            "Split %",
            "Cannibalization Independence Date L0",
        ]

        # Iterate over each row in the cannibItem_filtered DataFrame
        for idx, intersection in cannibItem_filtered.iterrows():
            # Step 1: Filter InitiativeLevels DataFrame for a unique match
            initiative_filter = InitiativeLevels[
                (InitiativeLevels[data_obj] == intersection[data_obj])
                & (InitiativeLevels[initiative_col] == intersection[initiative_col])
                & (InitiativeLevels[version_col] == intersection[version_col])
            ]

            # Step 2: Ensure only one match is found
            if initiative_filter.shape[0] != 1:
                raise ValueError(
                    f"Expected exactly one match in InitiativeLevels for row {idx}, found {initiative_filter.shape[0]}"
                )

                # Step 3: Drop unnecessary columns for efficient mapping
            initiative_filter_row = initiative_filter.drop(
                columns=[version_col, data_obj, initiative_col]
            ).iloc[0]

            # Create an empty dictionary to store the column mapping dynamically
            renamed_columns = {}

            for npi_col, cannib_col in zip(npi_level_columns, cannib_columns):
                # Extract the corresponding value from the initiative_filter_row
                value = initiative_filter_row[npi_col]
                # Rename the column to include the value in brackets
                renamed_columns[cannib_col] = f"{cannib_col.split('[')[0]}[{value}]"

            # Step 4: Rename the columns in the current row of cannibItem_filtered
            intersection_renamed = intersection.rename(renamed_columns)

            if isinstance(intersection_renamed, pd.Series):
                intersection_renamed = intersection_renamed.to_frame().T

            (
                key_item_col,
                key_account_col,
                key_channel_col,
                key_region_col,
                key_pnl_col,
                key_dd_col,
                key_loc_col,
            ) = get_key_columns(intersection_renamed)

            # Remove duplicate cols
            ItemMaster_req = ItemMaster[[pl_item_col, key_item_col]].drop_duplicates()
            AccountMaster_req = AccountMaster[
                [planning_account_col, key_account_col]
            ].drop_duplicates()
            ChannelMaster_req = ChannelMaster[
                [planning_channel_col, key_channel_col]
            ].drop_duplicates()
            RegionMaster_req = RegionMaster[[planning_region_col, key_region_col]].drop_duplicates()
            PnLMaster_req = PnLMaster[[planning_pnl_col, key_pnl_col]].drop_duplicates()
            DemandDomainMaster_req = DemandDomainMaster[
                [planning_demand_domain_col, key_dd_col]
            ].drop_duplicates()
            LocationMaster_req = LocationMaster[
                [planning_location_col, key_loc_col]
            ].drop_duplicates()

            # Merge with the assortment to get all the required columns
            Master_list = [
                (key_item_col, ItemMaster_req),
                (key_account_col, AccountMaster_req),
                (key_channel_col, ChannelMaster_req),
                (key_region_col, RegionMaster_req),
                (key_pnl_col, PnLMaster_req),
                (key_dd_col, DemandDomainMaster_req),
                (key_loc_col, LocationMaster_req),
            ]

            planning_lvel_df = reduce(merge_two, Master_list, intersection_renamed)
            planning_lvl_cols = [
                pl_item_col,
                planning_account_col,
                planning_channel_col,
                planning_demand_domain_col,
                planning_pnl_col,
                pl_loc_col,
                planning_region_col,
            ]

            AssortmentFinal_unique = AssortmentFinal[planning_lvl_cols + ["Assortment Final"]]
            # Ensure "Assortment Final" column is retained by merging it separately
            planning_lvel_df = planning_lvel_df.merge(
                AssortmentFinal_unique,
                on=planning_lvl_cols,  # Merge only on the required key columns
                how="left",
            )

            planning_lvel_df = planning_lvel_df.dropna(subset=["Assortment Final"])
            planning_lvel_df = planning_lvel_df.drop(columns=[from_item_col], errors="ignore")

            key_level_cols = [
                key_item_col,
                key_account_col,
                key_channel_col,
                key_region_col,
                key_pnl_col,
                key_dd_col,
                key_loc_col,
            ]

            splits_df = process_assortment_npi(planning_lvel_df, key_level_cols, key_cols_to_retain)

            # if split is empty we will assign new cols named Split%
            if splits_df.empty:
                splits_df["Split %"] = pd.Series(dtype=float)
            all_splits.append(splits_df)

        Splits = pd.concat(all_splits, ignore_index=True) if all_splits else pd.DataFrame()
        # Renaming the columns
        rename_mapping = {
            assortment_final: assortmentfinal_col_name,
            cannib_independence_dt_col: independence_dt_pllvl_output_name,
            "Split %": split_col_name,
        }

        Splits = Splits.rename(
            columns={k: v for k, v in rename_mapping.items() if k in Splits.columns}
        )
        Splits = Splits.sort_values(by=independence_dt_pllvl_output_name, ascending=True)

        Splits = Splits[cols_req_in_output_split]

        logger.info("Successfully executed {}".format(plugin_name))

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        cannibalizedForecast = pd.DataFrame(columns=cols_req_in_output)
        Cannib_Independence_Date = pd.DataFrame(columns=cols_req_in_output_independence)
        Splits = pd.DataFrame(columns=cols_req_in_output_split)

    return cannibalizedForecast, Cannibalization_Independence_Date, Splits
