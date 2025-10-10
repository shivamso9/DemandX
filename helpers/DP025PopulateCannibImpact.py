"""Populate Cannibalization Impact.

Return Cannibalized Forecast and Independance date.
"""

import datetime
import logging

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from o9Reference.common_utils.decorators import map_output_columns_to_dtypes
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

pd.options.display.max_rows = 25
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


logger = logging.getLogger("o9_logger")


col_mapping = {
    "600 Cannibalization.[Cannib Impact]": float,
    "600 Cannibalization.[Cannib Profile]": float,
    "Cannibalization_Independence_Date.[Cannibalization Independence Date]": "datetime64[ns]",
}


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


def to_date_time(df, target_col, source_col, format="%Y-%m-%d"):
    """Convert date time optimally."""
    df[target_col] = df[source_col].astype("string[pyarrow]")
    df[target_col] = pd.to_datetime(df[target_col], format=format, cache=True)
    return df


def merge_on_common_col(df1, df2, how, rhs_cols):
    """Merge 2 dfs based on common col.

    Parameters
    ----------
    df1 : lhs df
    df2 : rhs df
    how : inner/outer/left/right/etc
    rhs_cols : df2 columns copy to df1

    Output
    ----------
    return df1 after merging with df2 including rhs_cols
    """
    common_cols = list(set((set(df1.columns) & set(df2.columns))) - set([rhs_cols]))

    df2 = df2.groupby(common_cols)[rhs_cols].max().reset_index()

    df1 = pd.merge(df1, df2, on=common_cols, how=how)

    if rhs_cols not in df1.columns:
        df1[rhs_cols] = df1[rhs_cols + "_x"].where(
            pd.notna(df1[rhs_cols + "_x"]), df1[rhs_cols + "_y"]
        )
        df1.drop([rhs_cols + "_x", rhs_cols + "_y"], axis=1, inplace=True)

    return df1


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    ForecastData,
    TotalCombinations,
    NPIForecast,
    SelectedNewItemCustomerCombination,
    DefaultProfile,
    Parameters,
    TimeDimension,
    df_keys,
    PlanningMonthFormat=None,
):
    """DP025PopulateCannibImpact plugin."""
    plugin_name = "DP025PopulateCannibImpact"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # output measures
    cols_req_in_output = [
        "Version.[Version Name]",
        "from.[Account].[Planning Account]",
        "from.[PnL].[Planning PnL]",
        "from.[Region].[Planning Region]",
        "from.[Channel].[Planning Channel]",
        "from.[Location].[Planning Location]",
        "to.[Time].[Partial Week]",
        "to.[Item].[Planning Item]",
        "from.[Item].[Planning Item]",
        "600 Cannibalization.[Cannib Impact]",
        "600 Cannibalization.[Cannib Profile]",
        # "Cannibalization Independence Date"
    ]
    try:

        cannibalizedForecast = pd.DataFrame(columns=cols_req_in_output)

        if (
            len(DefaultProfile) == 0
            or len(NPIForecast) == 0
            or len(Parameters) == 0
            or len(ForecastData) == 0
            or len(TotalCombinations) == 0
            or len(SelectedNewItemCustomerCombination) == 0
        ):
            logger.warning("One of the inputs is Null. Exiting : {} ...".format(df_keys))
            logger.warning("Returning empty dataframe for this slice")
            return cannibalizedForecast

        logger.info("Selecting required columns from input dataframes ...")

        req_cols = [
            "Version.[Version Name]",
            "PLC Profile.[PLC Profile]",
            "PLC Profile.[PLC Time Bucket]",
            "Lifecycle Time.[Lifecycle Bucket]",
            "Default Profile",
        ]
        DefaultProfile = DefaultProfile[
            DefaultProfile["PLC Profile.[PLC Profile Type]"] == "Cannibalization Profile"
        ]
        DefaultProfile = DefaultProfile[req_cols]

        req_cols = [
            "Item.[Planning Item]",
            "Account.[Planning Account]",
            "PnL.[Planning PnL]",
            "Region.[Planning Region]",
            "Channel.[Planning Channel]",
            "Demand Domain.[Planning Demand Domain]",
            "Location.[Planning Location]",
            "Version.[Version Name]",
            "User Defined Cannibalization Period",
            "Intro Date",
            "Cannibalization Profile",
            "Cannibalization Profile Bucket",
        ]
        Parameters = Parameters[req_cols]

        req_cols = [
            "from.[Item].[Planning Item]",
            "from.[Account].[Planning Account]",
            "from.[PnL].[Planning PnL]",
            "from.[Region].[Planning Region]",
            "from.[Channel].[Planning Channel]",
            "from.[Location].[Planning Location]",
            "to.[Item].[Planning Item]",
            "Version.[Version Name]",
        ]
        TotalCombinations = TotalCombinations[req_cols]

        # Configurables
        item_col = "Item.[Planning Item]"
        location_col = "Location.[Planning Location]"
        account_col = "Account.[Planning Account]"
        region_col = "Region.[Planning Region]"
        demand_domain_col = "Demand Domain.[Planning Demand Domain]"
        pnl_col = "PnL.[Planning PnL]"
        channel_col = "Channel.[Planning Channel]"

        from_item_col = "from.[Item].[Planning Item]"
        from_location_col = "from.[Location].[Planning Location]"
        from_account_col = "from.[Account].[Planning Account]"
        from_region_col = "from.[Region].[Planning Region]"
        from_pnl_col = "from.[PnL].[Planning PnL]"
        from_channel_col = "from.[Channel].[Planning Channel]"
        to_item_col = "to.[Item].[Planning Item]"
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
        cannib_profile = "600 Cannibalization.[Cannib Profile]"
        cannib_impact = "600 Cannibalization.[Cannib Impact]"

        SelectedNewItemCustomerCombination = SelectedNewItemCustomerCombination[
            SelectedNewItemCustomerCombination["Populate Cannibalization Impact Assortment"] == 1
        ]

        Cannibalization_Independence_Date = SelectedNewItemCustomerCombination.copy()

        Parameters = pd.merge(
            SelectedNewItemCustomerCombination,
            Parameters,
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
            ],
        )
        Parameters.drop(
            columns="Populate Cannibalization Impact Assortment",
            axis=1,
            inplace=True,
        )

        Parameters["Intro Date"] = pd.to_datetime(Parameters["Intro Date"])

        TimeDimension = to_date_time(TimeDimension, day_col, daykey_col)

        TimeDimension = to_date_time(TimeDimension, month_col, monthkey_col)

        DefaultProfile["Default Profile"] = DefaultProfile["Default Profile"].fillna(0)

        B00Params = pd.merge(
            Parameters,
            DefaultProfile,
            how="inner",
            left_on=[
                version_col,
                "Cannibalization Profile",
                "Cannibalization Profile Bucket",
            ],
            right_on=[
                version_col,
                "PLC Profile.[PLC Profile]",
                "PLC Profile.[PLC Time Bucket]",
            ],
        )
        B00Params.drop(
            ["PLC Profile.[PLC Time Bucket]", "PLC Profile.[PLC Profile]"],
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
                "Intro Date",
                "Cannibalization Profile",
                "Cannibalization Profile Bucket",
                "User Defined Cannibalization Period",
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
                "Intro Date",
                "Cannibalization Profile",
                "Cannibalization Profile Bucket",
                "User Defined Cannibalization Period",
            ],
            observed=False,
        ).cumcount()

        # Divide based on the Cannibalization Profile Bucket
        Bucket_set = set(Parameters["Cannibalization Profile Bucket"].tolist())
        Final_df = pd.DataFrame()

        # Ensure 'Intro Date' is datetime64[ns] in both DataFrames
        Parameters["Intro Date"] = pd.to_datetime(Parameters["Intro Date"])

        # Removing timestamp from intro date since it needs to be merged with TimeDimension
        B00Params["Intro Date"] = pd.to_datetime(B00Params["Intro Date"], format="%d-%b-%Y")
        B00Params["Intro Date"] = B00Params["Intro Date"].dt.normalize()

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
            TimeDimension = to_date_time(
                TimeDimension,
                planning_month_col,
                planning_month_col,
                format=PlanningMonthFormat,
            )

            PlanningMonth_set = sorted(list(set(TimeDimension[planning_month_col].tolist())))
            PMParams = B00Params[B00Params["Cannibalization Profile Bucket"] == "Planning Month"]
            PMParams = pd.merge(
                PMParams,
                TimeDimension[[day_col, planning_month_col]],
                left_on="Intro Date",
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

            # Get unique combinations with all intersections
            unique_combinations = PMParams.drop_duplicates(
                subset=[
                    "Region.[Planning Region]",
                    "Item.[Planning Item]",
                    "PnL.[Planning PnL]",
                    "Location.[Planning Location]",
                    "Channel.[Planning Channel]",
                    "Demand Domain.[Planning Demand Domain]",
                    "Account.[Planning Account]",
                    "Intro Date",
                    "User Defined Cannibalization Period",
                    "Cannibalization Profile Bucket",
                ]
            )

            # Calculate independence date only for unique combinations
            unique_combinations["Cannibalization Independence Date"] = unique_combinations.apply(
                lambda row: calculate_cannibalization_independence_date(
                    row["Intro Date"],
                    row["User Defined Cannibalization Period"],
                    row["Cannibalization Profile Bucket"],
                    TimeDimension,
                    PlanningMonthFormat,
                ),
                axis=1,
            )

            # Copy independence date directly to Cannibalization_Independence_Date
            Cannibalization_Independence_Date = merge_on_common_col(
                Cannibalization_Independence_Date,
                unique_combinations,
                "left",
                "Cannibalization Independence Date",
            )

            # Apply the custom function to each group for offset calculation
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
                        "Intro Date",
                        planning_month_col,
                        "Cannibalization Profile",
                        "Cannibalization Profile Bucket",
                        "User Defined Cannibalization Period",
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
                    "Intro Date",
                    "Cannibalization Profile",
                    "Cannibalization Profile Bucket",
                    "User Defined Cannibalization Period",
                    "Offset",
                ]
            )
            PMParams = pd.merge(
                PMParams,
                TimeDimension[[partial_week_col, planning_month_col]].drop_duplicates(),
                how="inner",
                left_on=["Offset"],
                right_on=[planning_month_col],
            )
            PMParams.reset_index(drop=True, inplace=True)
            PMParams.drop(
                columns=[
                    "User Defined Cannibalization Period",
                    "Intro Date",
                    "Cannibalization Profile",
                    "Cannibalization Profile Bucket",
                    "Time.[Day]",
                    "Time.[Planning Month]_x",
                    "Time.[Planning Month]_y",
                    "Offset",
                ],
                axis=1,
                inplace=True,
            )
            Final_df = pd.concat([Final_df, PMParams], ignore_index=True)

        if "Month" in Bucket_set:
            Months_set = sorted(list(set(TimeDimension[month_col].tolist())))
            MonthParams = B00Params[B00Params["Cannibalization Profile Bucket"] == "Month"]
            MonthParams = pd.merge(
                MonthParams,
                TimeDimension[[day_col, month_col]],
                left_on="Intro Date",
                right_on=day_col,
                how="left",
            )

            # Get unique combinations with all intersections for Month bucket
            unique_month_combinations = MonthParams.drop_duplicates(
                subset=[
                    "Region.[Planning Region]",
                    "Item.[Planning Item]",
                    "PnL.[Planning PnL]",
                    "Location.[Planning Location]",
                    "Channel.[Planning Channel]",
                    "Demand Domain.[Planning Demand Domain]",
                    "Account.[Planning Account]",
                    "Intro Date",
                    "User Defined Cannibalization Period",
                    "Cannibalization Profile Bucket",
                ]
            )

            # Calculate independence date only for unique combinations
            unique_month_combinations["Cannibalization Independence Date"] = (
                unique_month_combinations.apply(
                    lambda row: calculate_cannibalization_independence_date(
                        row["Intro Date"],
                        row["User Defined Cannibalization Period"],
                        row["Cannibalization Profile Bucket"],
                        TimeDimension,
                        PlanningMonthFormat,
                    ),
                    axis=1,
                )
            )

            # Copy independence date directly to Cannibalization_Independence_Date
            Cannibalization_Independence_Date = merge_on_common_col(
                Cannibalization_Independence_Date,
                unique_month_combinations,
                "left",
                "Cannibalization Independence Date",
            )

            # Custom function to replace offsets with months
            def replace_offsets_with_months(group, months):
                starting_plan = group[month_col].iloc[0]
                start_index = months.index(starting_plan)
                required_size = len(months) - start_index
                group = group.head(required_size)
                plans_cycle = months[start_index:] + months[:start_index]
                group["Offset"] = [plans_cycle[i % len(plans_cycle)] for i in group["Offset"]]
                return group

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
                    "Intro Date",
                    month_col,
                    "Cannibalization Profile",
                    "Cannibalization Profile Bucket",
                    "User Defined Cannibalization Period",
                ]
            ).apply(replace_offsets_with_months, months=Months_set)
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
                    "Intro Date",
                    "Cannibalization Profile",
                    "Cannibalization Profile Bucket",
                    "User Defined Cannibalization Period",
                    month_col,
                ]
            )

            # Convert 'Time.[Month]' to datetime
            TimeDimension = to_date_time(TimeDimension, month_col, monthkey_col)

            # Merge MonthParams and TimeDimension on month to add partial weeks to MonthParams
            MonthParams = pd.merge(
                MonthParams,
                TimeDimension[[partial_week_col, month_col]].drop_duplicates(),
                how="inner",
                left_on=["Offset"],
                right_on=[month_col],
            )
            MonthParams.reset_index(drop=True, inplace=True)
            MonthParams.drop(
                columns=[
                    "User Defined Cannibalization Period",
                    "Intro Date",
                    "Cannibalization Profile",
                    "Cannibalization Profile Bucket",
                    "Time.[Month]_x",
                    "Time.[Month]_y",
                    "Offset",
                ],
                axis=1,
                inplace=True,
            )
            Final_df = pd.concat([Final_df, MonthParams], ignore_index=True)

        if "Week" in Bucket_set:
            TimeDimension[week_col] = TimeDimension[week_col].apply(
                lambda x: datetime.datetime.strptime(x, "%d-%b-%y")
            )
            Weeks_set = sorted(list(set(TimeDimension[week_col].tolist())))
            WeekParams = B00Params[B00Params["Cannibalization Profile Bucket"] == "Week"]
            WeekParams = pd.merge(
                WeekParams,
                TimeDimension[[day_col, week_col]],
                left_on="Intro Date",
                right_on=day_col,
                how="left",
            )

            # Get unique combinations with all intersections for Week bucket
            unique_week_combinations = WeekParams.drop_duplicates(
                subset=[
                    region_col,
                    "Item.[Planning Item]",
                    "PnL.[Planning PnL]",
                    "Location.[Planning Location]",
                    "Channel.[Planning Channel]",
                    "Demand Domain.[Planning Demand Domain]",
                    "Account.[Planning Account]",
                    "Intro Date",
                    "User Defined Cannibalization Period",
                    "Cannibalization Profile Bucket",
                ]
            )

            # Calculate independence date only for unique combinations
            unique_week_combinations["Cannibalization Independence Date"] = (
                unique_week_combinations.apply(
                    lambda row: calculate_cannibalization_independence_date(
                        row["Intro Date"],
                        row["User Defined Cannibalization Period"],
                        row["Cannibalization Profile Bucket"],
                        TimeDimension,
                        PlanningMonthFormat,
                    ),
                    axis=1,
                )
            )

            # Copy independence date directly to Cannibalization_Independence_Date
            Cannibalization_Independence_Date = merge_on_common_col(
                Cannibalization_Independence_Date,
                unique_week_combinations,
                "left",
                "Cannibalization Independence Date",
            )

            # Custom function to replace offsets with weeks
            def replace_offsets_with_weeks(group, weeks):
                starting_plan = group[week_col].iloc[0]
                start_index = weeks.index(starting_plan)
                required_size = len(weeks) - start_index
                group = group.head(required_size)
                plans_cycle = weeks[start_index:] + weeks[:start_index]
                group["Offset"] = [plans_cycle[i % len(plans_cycle)] for i in group["Offset"]]
                return group

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
                    "Intro Date",
                    week_col,
                    "Cannibalization Profile",
                    "Cannibalization Profile Bucket",
                    "User Defined Cannibalization Period",
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
                    "Intro Date",
                    "Cannibalization Profile",
                    "Cannibalization Profile Bucket",
                    "User Defined Cannibalization Period",
                    "Offset",
                ]
            )

            # Convert 'Time.[Week]' to datetime64[ns]
            TimeDimension = to_date_time(TimeDimension, week_col, weekkey_col)

            WeekParams = pd.merge(
                WeekParams,
                TimeDimension[[partial_week_col, week_col]].drop_duplicates(),
                how="inner",
                left_on=["Offset"],
                right_on=[week_col],
            )
            WeekParams.reset_index(drop=True, inplace=True)
            WeekParams.drop(
                columns=[
                    "User Defined Cannibalization Period",
                    "Intro Date",
                    "Cannibalization Profile",
                    "Cannibalization Profile Bucket",
                    "Time.[Day]",
                    "Time.[Week]_x",
                    "Time.[Week]_y",
                    "Offset",
                ],
                axis=1,
                inplace=True,
            )
            Final_df = pd.concat([Final_df, WeekParams], ignore_index=True)

        Final_df = pd.merge(
            NPIForecast,
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
        Final_df[cannib_impact] = Final_df["NPI Fcst Final"] * Final_df[cannib_profile]

        Final_df.drop(columns=["NPI Fcst Final", demand_domain_col], axis=1, inplace=True)
        Final_df.rename(
            columns={
                item_col: from_item_col,
                account_col: from_account_col,
                channel_col: from_channel_col,
                region_col: from_region_col,
                pnl_col: from_pnl_col,
                location_col: from_location_col,
                partial_week_col: to_partial_week_col,
            },
            inplace=True,
        )

        filtered_df = TotalCombinations.groupby(
            [
                version_col,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
                from_item_col,
            ],
            observed=False,
        ).filter(lambda x: x[to_item_col].nunique() > 1)
        filtered_df = pd.merge(
            filtered_df,
            Final_df,
            how="inner",
            on=[
                version_col,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
                from_item_col,
            ],
        )
        filtered_df = pd.merge(
            filtered_df,
            ForecastData,
            how="inner",
            left_on=[
                version_col,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
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
                item_col,
                partial_week_col,
            ],
            axis=1,
            inplace=True,
        )
        filtered_df["sum"] = filtered_df.groupby(
            [
                version_col,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
                from_item_col,
                to_partial_week_col,
                cannib_impact,
                cannib_profile,
            ],
            observed=False,
        )["Stat Fcst NPI BB"].transform("sum")
        filtered_df["CImpact"] = (
            filtered_df["Stat Fcst NPI BB"] * filtered_df[cannib_impact]
        ) / filtered_df["sum"]
        filtered_df.drop(
            columns=[
                "sum",
                "Stat Fcst NPI BB",
                cannib_impact,
                demand_domain_col,
            ],
            axis=1,
            inplace=True,
        )
        filtered_df.rename(columns={"CImpact": cannib_impact}, inplace=True)

        Final_df = pd.merge(
            TotalCombinations,
            Final_df,
            how="inner",
            on=[
                version_col,
                from_account_col,
                from_channel_col,
                from_region_col,
                from_pnl_col,
                from_location_col,
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

        # Remove
        Cannibalization_Independence_Date.drop(
            columns=["Populate Cannibalization Impact Assortment"],
            inplace=True,
        )
        Cannibalization_Independence_Date["Cannibalization Independence Date"] = pd.to_datetime(
            Cannibalization_Independence_Date["Cannibalization Independence Date"]
        )

        logger.info("Successfully executed {}".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        cannibalizedForecast = pd.DataFrame(columns=cols_req_in_output)

    return cannibalizedForecast, Cannibalization_Independence_Date
