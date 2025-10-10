import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.common_utils import (
    get_last_time_period,
    get_n_time_periods,
)
from o9Reference.common_utils.dataframe_utils import create_cartesian_product
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.stat_utils.time_series import calculate_trend

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None
logger = logging.getLogger("o9_logger")

col_mapping = {
    "Rate of Change Planning Item": float,
    "Rate of Change L4": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Actual_PlanningItem,
    PhaseOutHistoryPeriod,
    Actual_L4,
    TimeDimension,
    CurrentTimePeriod,
):
    plugin_name = "DP035DelistRecommendation"
    logger.info("Executing {}  ...".format(plugin_name))

    # Configurables
    version_col = "Version.[Version Name]"
    planning_region_col = "Region.[Planning Region]"
    planning_account_col = "Account.[Planning Account]"
    planning_channel_col = "Channel.[Planning Channel]"
    planning_pnl_col = "PnL.[Planning PnL]"
    planning_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    planning_location_col = "Location.[Planning Location]"
    planning_item_col = "Item.[Planning Item]"
    rate_of_change_of_pl_item_col = "Rate of Change Planning Item"
    l4_col = "Item.[L4]"
    rate_of_change_of_l4_col = "Rate of Change L4"
    phase_out_period_col = "Phase Out History Period (in months)"
    actual_col = "Actual"
    avg_sales_col = "Avg Sales"
    trend_col = "trend"
    partial_week_col = "Time.[Partial Week]"
    month_col = "Time.[Month]"
    month_key_col = "Time.[MonthKey]"

    RateOfChange_cols = [
        version_col,
        planning_channel_col,
        planning_location_col,
        planning_account_col,
        planning_demand_domain_col,
        planning_pnl_col,
        planning_region_col,
        planning_item_col,
        rate_of_change_of_pl_item_col,
        rate_of_change_of_l4_col,
    ]
    RateOfChange = pd.DataFrame(columns=RateOfChange_cols)
    try:
        if len(Actual_PlanningItem) == 0 or len(Actual_L4) == 0:
            logger.warning("Actual_PlanningItem/Actual_L4 is empty, returning empty dataframe ...")
            return RateOfChange

        logger.info("Joining actuals with time mapping ...")

        # join both actuals with time mapping
        Actual_L4_with_time_mapping = Actual_L4.merge(
            TimeDimension, on=partial_week_col, how="inner"
        )
        Actual_PlanningItem_with_time_mapping = Actual_PlanningItem.merge(
            TimeDimension, on=partial_week_col, how="inner"
        )

        if len(PhaseOutHistoryPeriod) == 0:
            logger.warning("Phase out history periods is empty, will return empty dataframe ...")
            return RateOfChange

        # get phase out period months
        phase_out_period = int(PhaseOutHistoryPeriod[phase_out_period_col].unique()[0])
        logger.info(f"phase_out_period : {phase_out_period}")

        relevant_time_name = month_col
        relevant_time_key = month_key_col

        logger.info(f"relevant_time_name : {relevant_time_name}")
        logger.info(f"relevant_time_key : {relevant_time_key}")

        # collect relevant time mapping
        relevant_time_mapping = (
            TimeDimension[[relevant_time_name, relevant_time_key]]
            .drop_duplicates()
            .sort_values(relevant_time_key)
        )

        # collect last time period
        latest_time_name = get_last_time_period(
            CurrentTimePeriod,
            TimeDimension,
            relevant_time_name,
            relevant_time_key,
        )
        logger.info(f"latest_time_name : {latest_time_name}")

        # note the negative sign to phase_out_period
        phase_out_periods = get_n_time_periods(
            latest_value=latest_time_name,
            periods=-phase_out_period,
            time_mapping=relevant_time_mapping,
            time_attribute={relevant_time_name: relevant_time_key},
            include_latest_value=True,
        )
        logger.info(f"phase_out_periods : {phase_out_periods}")

        # get the phase out period keys as well
        phase_out_period_keys = list(
            relevant_time_mapping[
                relevant_time_mapping[relevant_time_name].isin(phase_out_periods)
            ][relevant_time_key].unique()
        )

        # collect data for phase out periods
        Actual_L4_phase_out = Actual_L4_with_time_mapping[
            Actual_L4_with_time_mapping[relevant_time_name].isin(phase_out_periods)
        ]

        Actual_PlanningItem_phase_out = Actual_PlanningItem_with_time_mapping[
            Actual_PlanningItem_with_time_mapping[relevant_time_name].isin(phase_out_periods)
        ]

        # groupby relevant fields and sum history measure
        cols_to_groupby = [
            version_col,
            planning_channel_col,
            planning_location_col,
            planning_account_col,
            planning_demand_domain_col,
            planning_pnl_col,
            planning_region_col,
            planning_item_col,
            l4_col,
            relevant_time_key,
        ]
        logger.info(f"grouping by {cols_to_groupby} and aggregating {actual_col} ...")

        Actual_PlanningItem_phase_out = Actual_PlanningItem_phase_out.groupby(
            cols_to_groupby
        ).sum()[[actual_col]]
        Actual_PlanningItem_phase_out.reset_index(inplace=True)
        Actual_PlanningItem_phase_out.sort_values(cols_to_groupby, inplace=True)

        # remove planning item
        cols_to_groupby.remove(planning_item_col)
        logger.info(f"grouping by {cols_to_groupby} and aggregating {actual_col} ...")

        Actual_L4_phase_out = Actual_L4_phase_out.groupby(cols_to_groupby).sum()[[actual_col]]
        Actual_L4_phase_out.reset_index(inplace=True)
        Actual_L4_phase_out.sort_values(cols_to_groupby, inplace=True)

        logger.info("Creating cartesian product with time ...")
        # create cartesian product
        PlanningItem_master = Actual_PlanningItem_phase_out[
            [
                version_col,
                planning_channel_col,
                planning_location_col,
                planning_account_col,
                planning_demand_domain_col,
                planning_pnl_col,
                planning_region_col,
                planning_item_col,
                l4_col,
            ]
        ].drop_duplicates()

        PlanningItem_master_with_time = create_cartesian_product(
            df1=PlanningItem_master,
            df2=pd.DataFrame({relevant_time_key: phase_out_period_keys}),
        )

        L4_master = Actual_L4_phase_out[
            [
                version_col,
                planning_channel_col,
                planning_location_col,
                planning_account_col,
                planning_demand_domain_col,
                planning_pnl_col,
                planning_region_col,
                l4_col,
            ]
        ].drop_duplicates()

        L4_master_with_time = create_cartesian_product(
            df1=L4_master,
            df2=pd.DataFrame({relevant_time_key: phase_out_period_keys}),
        )

        logger.info(f"Actual_PlanningItem_phase_out, shape : {Actual_PlanningItem_phase_out.shape}")

        # do left joins
        Actual_PlanningItem_nas_filled = PlanningItem_master_with_time.merge(
            Actual_PlanningItem_phase_out,
            how="left",
            on=[
                version_col,
                planning_channel_col,
                planning_location_col,
                planning_account_col,
                planning_demand_domain_col,
                planning_pnl_col,
                planning_region_col,
                planning_item_col,
                l4_col,
                relevant_time_key,
            ],
        )
        # fill na with zeros
        Actual_PlanningItem_nas_filled[actual_col].fillna(0, inplace=True)
        logger.info(
            f"Actual_PlanningItem_nas_filled, shape : {Actual_PlanningItem_nas_filled.shape}"
        )

        logger.info(f"Actual_L4_phase_out, shape : {Actual_L4_phase_out.shape}")
        Actual_L4_nas_filled = L4_master_with_time.merge(
            Actual_L4_phase_out,
            how="left",
            on=[
                version_col,
                planning_channel_col,
                planning_location_col,
                planning_account_col,
                planning_demand_domain_col,
                planning_pnl_col,
                planning_region_col,
                l4_col,
                relevant_time_key,
            ],
        )
        # fill na with zeros
        Actual_L4_nas_filled[actual_col].fillna(0, inplace=True)
        logger.info(f"Actual_L4_nas_filled, shape : {Actual_L4_nas_filled.shape}")

        logger.info("Creating pivot ...")
        # pivot data and calculate avg sales, trend
        pivot_Actual_PlanningItem = Actual_PlanningItem_nas_filled.pivot_table(
            actual_col,
            [
                version_col,
                planning_channel_col,
                planning_location_col,
                planning_account_col,
                planning_demand_domain_col,
                planning_pnl_col,
                planning_region_col,
                planning_item_col,
                l4_col,
            ],
            relevant_time_key,
        )
        logger.info(f"Calculating {avg_sales_col} and {trend_col} ...")
        pivot_Actual_PlanningItem[avg_sales_col] = pivot_Actual_PlanningItem.mean(axis=1)
        pivot_Actual_PlanningItem[trend_col] = pivot_Actual_PlanningItem.apply(
            lambda x: calculate_trend(x.to_numpy()), axis=1
        )
        pivot_Actual_PlanningItem[rate_of_change_of_pl_item_col] = np.where(
            pivot_Actual_PlanningItem[avg_sales_col] != 0,
            pivot_Actual_PlanningItem[trend_col] / pivot_Actual_PlanningItem[avg_sales_col],
            0,
        )

        pivot_Actual_L4 = Actual_L4_nas_filled.pivot_table(
            actual_col,
            [
                version_col,
                planning_channel_col,
                planning_location_col,
                planning_account_col,
                planning_demand_domain_col,
                planning_pnl_col,
                planning_region_col,
                l4_col,
            ],
            relevant_time_key,
        )
        pivot_Actual_L4[avg_sales_col] = pivot_Actual_L4.mean(axis=1)
        pivot_Actual_L4[trend_col] = pivot_Actual_L4.apply(
            lambda x: calculate_trend(x.to_numpy()), axis=1
        )
        pivot_Actual_L4[rate_of_change_of_l4_col] = np.where(
            pivot_Actual_L4[avg_sales_col] != 0,
            pivot_Actual_L4[trend_col] / pivot_Actual_L4[avg_sales_col],
            0,
        )

        # select required column
        pivot_Actual_PlanningItem = pivot_Actual_PlanningItem[[rate_of_change_of_pl_item_col]]

        # reset index for join
        pivot_Actual_PlanningItem.reset_index(inplace=True)

        # select required column
        pivot_Actual_L4 = pivot_Actual_L4[[rate_of_change_of_l4_col]]

        # reset index for join
        pivot_Actual_L4.reset_index(inplace=True)

        logger.info(
            f"Combining {rate_of_change_of_pl_item_col} and {rate_of_change_of_l4_col} into single output dataframe ..."
        )
        # join with l4 level values dataframe
        pivot_Actual_PlanningItem = pivot_Actual_PlanningItem.merge(
            pivot_Actual_L4,
            on=[
                version_col,
                planning_channel_col,
                planning_location_col,
                planning_account_col,
                planning_demand_domain_col,
                planning_pnl_col,
                planning_region_col,
                l4_col,
            ],
            how="inner",
        )

        # subset the relevant columns
        RateOfChange = pivot_Actual_PlanningItem[RateOfChange_cols]

        logger.info(f"Successfully executed {plugin_name} ...")

    except Exception as e:
        logger.exception("Exception {}".format(e))
        RateOfChange = pd.DataFrame(columns=RateOfChange_cols)
    return RateOfChange
