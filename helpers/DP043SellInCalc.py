import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants
from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

# DONE : Add the output columns and datatypes
col_mapping = {
    "Total Supply": float,
    "Sell In Calc": float,
    "Ch INV L1": float,
    "Ch INV": float,
    "Sell In AM Offset": float,
    "Sell In AM": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Lead_Time_Grains: str,
    grains: str,
    InTransitInventory: pd.DataFrame,
    IncludePastInTransitInventory: pd.DataFrame,
    ConsiderLeadTime: pd.DataFrame,
    LeadTime: pd.DataFrame,
    CurrentWeek: pd.DataFrame,
    TimeDimension: pd.DataFrame,
    df_keys,
):
    # TODO : Add docstrings over here
    """Calculating Sell In Calc from Total Demand and Total Supply

    Args:
        Lead_Time_Grains (str): grains to be used for Lead Time Dataframe
        grains (str): grains to be used for Input Dataframe
        InTransitInventory (pd.DataFrame): Input Dataframe
        IncludePastInTransitInventory (pd.DataFrame): DataFrame that contains boolean values, indicating whether In Transit Inventory Agg should be included in the calculations
        ConsiderLeadTime (pd.DataFrame): DataFrame that contains boolean values, indicating whether Lead Time should be included in the calculations
        LeadTime (pd.DataFrame): Lead Time Dataframe
        CurrentWeek (pd.DataFrame): Current Week
        TimeDimension (pd.DataFrame): Time master data

    Raises:
        ValueError: If any of the required columns are missing.
        ValueError: If any of the required columns are missing.
        ValueError: If Time Dimension is empty.

    Returns:
        _type_: Dataframe
    """
    plugin_name = "DP043SellInCalc"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    WEEK_KEY = "Time.[WeekKey]"
    InTransitInventory_col = "In Transit Inventory"
    TotalDemand_col = "Total Demand"
    TotalSupply_col = "Total Supply"
    SellInCalc_col = "Sell In Calc"
    SellOut_col = "Sell Out"
    SellInCalcLT_col = "Sell In AM Offset"
    LeadTime_col = "Sell In Lead Time"
    ChINV_col = "Ch INV"
    ChINVL1_col = "Ch INV L1"
    IncludePastInTransitInventory_col = "Include Past In Transit Inventory"
    InTransitInventoryAgg_col = "Past In Transit Inventory"
    ConsiderLeadTime_col = "Consider Lead Time"
    SellInAM_col = "Sell In AM"
    SellInOverride_col = "Sell In Override"

    grains = get_list_of_grains_from_string(grains)

    Lead_Time_Grains = get_list_of_grains_from_string(Lead_Time_Grains)

    cols_required_in_output = (
        [o9Constants.VERSION_NAME]
        + grains
        + [o9Constants.WEEK]
        + [
            TotalSupply_col,
            SellInCalc_col,
            ChINVL1_col,
            ChINV_col,
            SellInCalcLT_col,
            SellInAM_col,
        ]
    )
    Output = pd.DataFrame(columns=cols_required_in_output)
    # DONE : use pre commit config for formatting (https://pre-commit.com/) -Done
    try:
        # Check if the Inputs consists of data
        if (
            InTransitInventory[SellOut_col].count() == 0
            or InTransitInventory[TotalDemand_col].count() == 0
            or TimeDimension.empty
        ):
            raise ValueError("One or more of the inputs are empty! Check logs/ inputs for error!")

        # Merging TimeDimension with InTransitInventoryInput
        logger.debug("Joining Input data with time_mapping ...")
        InTransitInventoryInput = pd.merge(
            InTransitInventory,
            TimeDimension[[o9Constants.WEEK, WEEK_KEY]],
            on=[o9Constants.WEEK],
            how="inner",
        )
        InTransitInventoryInput.drop_duplicates(inplace=True)

        # Calculating ChINVL1 from last week's ChINV
        # Grouping the inputs at grains Level
        logger.debug("Joining Input data at grains level ...")
        ChINVL1 = InTransitInventoryInput.groupby(grains)
        InTransitInventoryInput = []
        for group, group_df in ChINVL1:
            group_df.sort_values(by=WEEK_KEY, inplace=True)
            group_df.reset_index(drop=True, inplace=True)
            group_df[ChINVL1_col] = group_df[ChINV_col].shift(1)
            InTransitInventoryInput += [group_df]
        InTransitInventoryInput = concat_to_dataframe(InTransitInventoryInput)
        InTransitInventoryInput.drop_duplicates(inplace=True)
        InTransitInventoryInput.sort_values(by=WEEK_KEY, inplace=True)

        min_week_key = CurrentWeek[WEEK_KEY].min()

        # filtering data after current week
        InTransitInventoryInput = InTransitInventoryInput[
            InTransitInventoryInput[WEEK_KEY] >= min_week_key
        ]

        LeadTimeCols = [o9Constants.VERSION_NAME] + Lead_Time_Grains

        logger.debug(f"Lead Time grains are : {LeadTimeCols}")
        # Merging Dataframes LeadTime and InTransitInventoryInput
        logger.debug("Joining Input data with Lead Time_mapping ...")
        InTransitInventoryInput = InTransitInventoryInput.merge(
            LeadTime, on=LeadTimeCols, how="left"
        )
        InTransitInventoryInput[LeadTime_col] = InTransitInventoryInput[LeadTime_col].fillna(0)
        InTransitInventoryInput[InTransitInventory_col] = InTransitInventoryInput[
            InTransitInventory_col
        ].fillna(0)
        InTransitInventoryInput[InTransitInventoryAgg_col] = InTransitInventoryInput[
            InTransitInventoryAgg_col
        ].fillna(0)
        InTransitInventoryInput[SellOut_col] = InTransitInventoryInput[SellOut_col].fillna(0)
        InTransitInventoryInput[TotalDemand_col] = InTransitInventoryInput[TotalDemand_col].fillna(
            0
        )
        InTransitInventoryInput.loc[
            InTransitInventoryInput[WEEK_KEY] == min_week_key, ChINVL1_col
        ].fillna(0)

        inputlist = []
        intransitnew = []
        finalnew = []

        # Grouping the inputs at grains Level
        ChINV = InTransitInventoryInput.groupby(grains)
        for group, group_df3 in ChINV:
            group_df3[ChINVL1_col]
            SellInCalc = group_df3.groupby([WEEK_KEY])
            # Calculating ChINVL1 from last week's ChINV
            val = group_df3[group_df3[WEEK_KEY] == min_week_key]
            # DONE :use np. where for the below codes - done
            value = np.where(val[ChINVL1_col].notna(), val[ChINVL1_col].iloc[0], 0)

            for final, final_df in SellInCalc:
                # Calculating TotalSupply
                if final == min_week_key:
                    pass
                else:
                    final_df[ChINVL1_col] = value
                if (IncludePastInTransitInventory[IncludePastInTransitInventory_col] == 1).any():
                    final_df[TotalSupply_col] = (
                        final_df[ChINVL1_col]
                        + final_df[InTransitInventory_col]
                        + final_df[InTransitInventoryAgg_col]
                    )
                else:
                    final_df[TotalSupply_col] = (
                        final_df[ChINVL1_col] + final_df[InTransitInventory_col]
                    )
                # Calculating SellInCalc after frozen_period, if flag is true
                if (ConsiderLeadTime[ConsiderLeadTime_col] == 1).any():
                    group_df3 = group_df3.sort_values(by=WEEK_KEY)
                    frozen_period = int(final_df[LeadTime_col].iloc[0])
                    weeks = group_df3[WEEK_KEY].drop_duplicates().to_list()
                    new_week = weeks[frozen_period]
                    final_df.loc[final_df[WEEK_KEY] >= new_week, SellInCalc_col] = (
                        final_df[TotalDemand_col] - final_df[TotalSupply_col]
                    )
                    final_df.loc[
                        final_df[SellInCalc_col] < 0,
                        SellInCalc_col,
                    ] = 0
                    # Checking whether Override is present
                    final_df[SellInAM_col] = np.where(
                        final_df.get(SellInOverride_col).notna(),
                        final_df[SellInOverride_col],
                        final_df[SellInCalc_col],
                    )
                    final_df.loc[final_df[WEEK_KEY] < new_week, ChINV_col] = (
                        final_df[TotalSupply_col] - final_df[SellOut_col]
                    )
                    final_df.loc[final_df[WEEK_KEY] >= new_week, ChINV_col] = (
                        final_df[SellInAM_col] + final_df[TotalSupply_col] - final_df[SellOut_col]
                    )
                    final_df.loc[final_df[ChINV_col] < 0, ChINV_col] = 0
                # Calculating SellInCalc for all time periods
                else:
                    final_df[SellInCalc_col] = final_df[TotalDemand_col] - final_df[TotalSupply_col]
                    final_df.loc[
                        final_df[TotalDemand_col] < final_df[TotalSupply_col],
                        SellInCalc_col,
                    ] = 0
                    # Checking whether Override is present
                    final_df[SellInAM_col] = np.where(
                        final_df.get(SellInOverride_col).notna(),
                        final_df[SellInOverride_col],
                        final_df[SellInCalc_col],
                    )
                    final_df[ChINV_col] = (
                        final_df[SellInAM_col] + final_df[TotalSupply_col] - final_df[SellOut_col]
                    )
                    final_df.loc[final_df[ChINV_col] < 0, ChINV_col] = 0

                val1 = final_df[ChINV_col]

                # DONE  : use np.where for the below if else - done
                value = np.where(val1.notna(), val1.iloc[0], 0)

                if value < 0:
                    value = 0
                inputlist += [final_df]
            intransitnew = concat_to_dataframe(inputlist)
        finalnew += [intransitnew]
        finalone = concat_to_dataframe(finalnew)
        logger.debug("Calculating Sell In AM Offset ...")
        # Shifting SellInCalcLT with SellInAM by LeadTime if the ConsiderLeadTime flag is True
        if (ConsiderLeadTime[ConsiderLeadTime_col] == 1).any():
            new_df_list2 = []

            finalone = finalone.sort_values(by=WEEK_KEY)
            SellInCalcLT = finalone.groupby(grains)
            for group1, group_df1 in SellInCalcLT:
                frozen_period_new = int(group_df1[LeadTime_col].iloc[0])
                group_df1[SellInCalcLT_col] = group_df1[SellInAM_col].shift(-frozen_period_new)
                new_df_list2 += [group_df1]
            new_df = concat_to_dataframe(new_df_list2)

        else:
            cols_required_in_output = (
                [o9Constants.VERSION_NAME]
                + grains
                + [o9Constants.WEEK]
                + [
                    TotalSupply_col,
                    SellInCalc_col,
                    ChINVL1_col,
                    ChINV_col,
                    SellInAM_col,
                ]
            )

            new_df = finalone
        new_df
        Output = new_df[cols_required_in_output]

        # DONE : What is the use of this pass here? - removed
        # Your code ends here
        logger.info("Output created... Returning it to the caller fn")
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)
    return Output
