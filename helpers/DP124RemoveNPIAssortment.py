"""Delete Created Assortment Plugin for Flexible NPI."""

import logging
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
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


def user_selected_level(
    InitiativeLevel_required,
    initiative_col,
    npi_item_level_col,
    npi_account_level_col,
    npi_channel_level_col,
    npi_region_level_col,
    npi_pnl_level_col,
    npi_dd_level_col,
    npi_location_level_col,
):
    """Return formatted columns of user selected level."""
    # Selecting the required columns
    InitiativeLevel_required = InitiativeLevel_required[
        [
            npi_item_level_col,
            npi_account_level_col,
            npi_channel_level_col,
            npi_region_level_col,
            npi_pnl_level_col,
            npi_dd_level_col,
            npi_location_level_col,
        ]
    ].drop_duplicates()

    # Formatting the cols name
    for col in InitiativeLevel_required.columns:
        if "NPI" in col and "Level" in col:
            attribute = col.split("NPI")[1].split("Level")[0].strip()
            InitiativeLevel_required[col] = (
                str(attribute) + ".[" + InitiativeLevel_required[col] + "]"
            )

    return list(InitiativeLevel_required.iloc[0])


def merge_two(df1, df2):
    """Merge two dfs."""
    common_cols = list(set(df1.columns) & set(df2.columns))
    return pd.merge(df1, df2, on=common_cols, how="inner")


col_mapping = {
    "NPI Fcst Published": float,
    "Assortment New": float,
    "Cannib Final Split %": float,
    "Cannib Override Split %": float,
    "Cannib System Split %": float,
    "Cannibalization Independence Date Planning Level": "datetime64[ns]",
    "NPI Final Split %": float,
    "NPI Override Split %": float,
    "NPI Planning Assortment by Level": float,
    "NPI System Split %": float,
    "NPI Assortment Final by Level": float,
    "NPI Assortment Final": float,
    "Planning Level Cannibalization Impact": float,
    "Planning Level NPI Fcst": float,
    "Planning Level Cannibalization Impact L1": float,
    "Planning Level NPI Fcst L1": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    RemoveAssortmentScope: Optional[str] = None,
    # Data
    NPIRemoveAssortmentFlag: pd.DataFrame = None,
    InitiativeLevel: pd.DataFrame = None,
    NPIFcstPublished: pd.DataFrame = None,
    AssortmentNew: pd.DataFrame = None,
    InitiativePlanningLevel: pd.DataFrame = None,
    InitiativeLevelAssortment: pd.DataFrame = None,
    InitiativeAssortment: pd.DataFrame = None,
    InitiativePlanningLevelNewProductForecast: pd.DataFrame = None,
    InitiativeNewProductForecast: pd.DataFrame = None,
    # Master data
    ItemMaster: pd.DataFrame = None,
    AccountMaster: pd.DataFrame = None,
    ChannelMaster: pd.DataFrame = None,
    RegionMaster: pd.DataFrame = None,
    PnLMaster: pd.DataFrame = None,
    DemandDomainMaster: pd.DataFrame = None,
    LocationMaster: pd.DataFrame = None,
    # Others
    df_keys: Optional[dict] = None,
):
    """Entry point of the script."""
    plugin_name = "DP124RemoveNPIAssortment"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    pl_item_col = "Item.[Planning Item]"
    item_col = "Item.[Item]"
    loc_col = "Location.[Location]"
    initiative_col = "Initiative.[Initiative]"
    data_object_col = "Data Object.[Data Object]"
    partial_week_col = "Time.[Partial Week]"

    planning_channel_col = "Channel.[Planning Channel]"
    planning_account_col = "Account.[Planning Account]"
    planning_pnl_col = "PnL.[Planning PnL]"
    planning_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    planning_region_col = "Region.[Planning Region]"
    planning_location_col = "Location.[Planning Location]"

    npi_item_col = "Item.[NPI Item]"
    npi_account_col = "Account.[NPI Account]"
    npi_channel_col = "Channel.[NPI Channel]"
    npi_region_col = "Region.[NPI Region]"
    npi_pnl_col = "PnL.[NPI PnL]"
    npi_location_col = "Location.[NPI Location]"
    npi_demand_domain_col = "Demand Domain.[NPI Demand Domain]"

    npi_assorment_final_by_level_col = "NPI Assortment Final by Level"
    npi_item_level_col = "NPI Item Level"
    npi_account_level_col = "NPI Account Level"
    npi_channel_level_col = "NPI Channel Level"
    npi_region_level_col = "NPI Region Level"
    npi_pnl_level_col = "NPI PnL Level"
    npi_location_level_col = "NPI Location Level"
    npi_dd_level_col = "NPI Demand Domain Level"
    npi_fcst_pulished_col = "NPI Fcst Published"
    assortment_new_col = "Assortment New"
    cannib_final_split_perc_col = "Cannib Final Split %"
    cannib_override_split_perc_col = "Cannib Override Split %"
    cannib_sys_split_perc_col = "Cannib System Split %"
    cannib_ind_date_pl_level_col = "Cannibalization Independence Date Planning Level"
    npi_final_split_perc_col = "NPI Final Split %"
    npi_override_split_perf_col = "NPI Override Split %"
    npi_planning_ass_by_level_col = "NPI Planning Assortment by Level"
    npi_sys_split_perc_col = "NPI System Split %"
    npi_ass_final_col = "NPI Assortment Final"
    pl_level_cann_impact_col = "Planning Level Cannibalization Impact"
    pl_level_npi_fcst_col = "Planning Level NPI Fcst"
    pl_level_cann_impact_l1_col = "Planning Level Cannibalization Impact L1"
    pl_level_npi_fcst_l1_col = "Planning Level NPI Fcst L1"

    # output columns
    cols_required_in_output_npi_fcst_pub = [
        version_col,
        pl_item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_location_col,
        planning_demand_domain_col,
        partial_week_col,
        npi_fcst_pulished_col,
    ]

    cols_required_in_output_ass_new = [
        version_col,
        item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        loc_col,
        assortment_new_col,
    ]

    cols_required_in_output_initiative_pl_level = [
        version_col,
        initiative_col,
        data_object_col,
        pl_item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        planning_location_col,
        cannib_final_split_perc_col,
        cannib_override_split_perc_col,
        cannib_sys_split_perc_col,
        cannib_ind_date_pl_level_col,
        npi_final_split_perc_col,
        npi_override_split_perf_col,
        npi_planning_ass_by_level_col,
        npi_sys_split_perc_col,
    ]

    cols_required_in_output_initiative_level_ass = [
        version_col,
        initiative_col,
        data_object_col,
        item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        loc_col,
        npi_assorment_final_by_level_col,
    ]

    cols_required_in_output_initiative_ass = [
        version_col,
        initiative_col,
        item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        loc_col,
        npi_ass_final_col,
    ]

    cols_required_in_output_initiative_pl_level_new_prod_fcst = [
        version_col,
        initiative_col,
        data_object_col,
        pl_item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        planning_location_col,
        partial_week_col,
        pl_level_cann_impact_col,
        pl_level_npi_fcst_col,
    ]

    cols_required_in_output_initiative_new_prod_fcst = [
        version_col,
        initiative_col,
        pl_item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        planning_location_col,
        partial_week_col,
        pl_level_cann_impact_l1_col,
        pl_level_npi_fcst_l1_col,
    ]

    # Output empty dataframes
    NPIFcstPublishedDeleted = pd.DataFrame(columns=cols_required_in_output_npi_fcst_pub)
    AssortmentNewDeleted = pd.DataFrame(columns=cols_required_in_output_ass_new)
    InitiativePlanningLevelDeleted = pd.DataFrame(
        columns=cols_required_in_output_initiative_pl_level
    )
    InitiativeLevelAssortmentDeleted = pd.DataFrame(
        columns=cols_required_in_output_initiative_level_ass
    )
    InitiativeAssortmentDeleted = pd.DataFrame(columns=cols_required_in_output_initiative_ass)
    InitiativePlanningLevelNewProdFcstDeleted = pd.DataFrame(
        columns=cols_required_in_output_initiative_pl_level_new_prod_fcst
    )
    InitiativeNewProductForecastDeleted = pd.DataFrame(
        columns=cols_required_in_output_initiative_new_prod_fcst
    )

    try:
        allowed_options = ["Full", "Partial"]
        if not RemoveAssortmentScope in allowed_options:
            logger.warning(
                f"RemoveAssortmentScope is invalid. User input: {RemoveAssortmentScope}. Options: {allowed_options}"
            )
            logger.info("Considering RemoveAssortmentScope as 'Partial'")
            RemoveAssortmentScope = "Partial"

        if len(NPIRemoveAssortmentFlag) == 0:
            logger.exception(f"No assortment data found for deletion for slice: {df_keys}...")

        # Iterate for each specific different levels

        NPIFcstPublishedDeleted_list = []
        AssortmentNewDeleted_list = []
        InitiativePlanningLevelDeleted_list = []
        InitiativeLevelAssortmentDeleted_list = []
        InitiativeAssortmentDeleted_list = []
        InitiativePlanningLevelNewProductForecastDeleted_list = []
        InitiativeNewProductForecastDeleted_list = []

        assorment_group = NPIRemoveAssortmentFlag.groupby([initiative_col, data_object_col])
        for _level, the_assortment in assorment_group:

            the_initiative = _level[0]
            the_level = _level[1]

            InitiativeLevel_required = InitiativeLevel[
                (InitiativeLevel[initiative_col] == the_initiative)
                & (InitiativeLevel[data_object_col] == the_level)
            ]
            if len(InitiativeLevel_required) == 0:
                logger.warning(
                    f"Does not found level for slice: {df_keys} for Initiative,Level: {_level} "
                )
                logger.warning("Skipping the intersection ...")
                continue

            # Get user defined level
            (
                user_item_level_col,
                user_account_level_col,
                user_channel_level_col,
                user_region_level_col,
                user_pnl_level_col,
                user_dd_level_col,
                user_loc_level_col,
            ) = user_selected_level(
                InitiativeLevel_required,
                initiative_col,
                npi_item_level_col,
                npi_account_level_col,
                npi_channel_level_col,
                npi_region_level_col,
                npi_pnl_level_col,
                npi_dd_level_col,
                npi_location_level_col,
            )

            # Rename to the belonging grain
            the_assortment.rename(
                columns={
                    npi_item_col: user_item_level_col,
                    npi_account_col: user_account_level_col,
                    npi_channel_col: user_channel_level_col,
                    npi_region_col: user_region_level_col,
                    npi_pnl_col: user_pnl_level_col,
                    npi_demand_domain_col: user_dd_level_col,
                    npi_location_col: user_loc_level_col,
                },
                inplace=True,
            )

            # Get all the levels
            master_list = [
                ItemMaster,
                AccountMaster,
                ChannelMaster,
                RegionMaster,
                PnLMaster,
                DemandDomainMaster,
                LocationMaster,
            ]
            the_assortment_w_master = reduce(merge_two, master_list, the_assortment)

            # ------------------- 1. Deleting NPIFcstPublished data
            key = [
                version_col,
                pl_item_col,
                planning_account_col,
                planning_channel_col,
                planning_region_col,
                planning_pnl_col,
                planning_location_col,
                planning_demand_domain_col,
            ]
            the_NPIFcstPublished_to_delete = pd.merge(
                NPIFcstPublished,
                the_assortment_w_master[key].drop_duplicates(),
                on=key,
                how="inner",
            )

            the_NPIFcstPublished_to_delete[npi_fcst_pulished_col] = np.nan

            # col req
            the_NPIFcstPublished_to_delete = the_NPIFcstPublished_to_delete[
                cols_required_in_output_npi_fcst_pub
            ]

            NPIFcstPublishedDeleted_list.append(the_NPIFcstPublished_to_delete)

            # ------------------- 2. Deleting AssortmentNew data
            key = [
                version_col,
                item_col,
                planning_account_col,
                planning_channel_col,
                planning_region_col,
                planning_pnl_col,
                loc_col,
                planning_demand_domain_col,
            ]
            the_AssortmentNew_to_delete = pd.merge(
                AssortmentNew, the_assortment_w_master[key].drop_duplicates(), how="inner"
            )

            the_AssortmentNew_to_delete[assortment_new_col] = np.nan

            # col req
            the_AssortmentNew_to_delete = the_AssortmentNew_to_delete[
                cols_required_in_output_ass_new
            ]

            AssortmentNewDeleted_list.append(the_AssortmentNew_to_delete)

            # ------------------- 3. Deleting InitiativePlanningLevel data
            if RemoveAssortmentScope == "Full":
                key = [
                    version_col,
                    initiative_col,
                    data_object_col,
                    pl_item_col,
                    planning_account_col,
                    planning_channel_col,
                    planning_region_col,
                    planning_pnl_col,
                    planning_demand_domain_col,
                    planning_location_col,
                ]
                the_InitiativePlanningLevel_to_delete = pd.merge(
                    InitiativePlanningLevel,
                    the_assortment_w_master[key].drop_duplicates(),
                    on=key,
                    how="inner",
                )

                the_InitiativePlanningLevel_to_delete[cannib_final_split_perc_col] = np.nan
                the_InitiativePlanningLevel_to_delete[cannib_override_split_perc_col] = np.nan
                the_InitiativePlanningLevel_to_delete[cannib_sys_split_perc_col] = np.nan
                the_InitiativePlanningLevel_to_delete[cannib_ind_date_pl_level_col] = np.nan
                the_InitiativePlanningLevel_to_delete[npi_final_split_perc_col] = np.nan
                the_InitiativePlanningLevel_to_delete[npi_override_split_perf_col] = np.nan
                the_InitiativePlanningLevel_to_delete[npi_planning_ass_by_level_col] = np.nan
                the_InitiativePlanningLevel_to_delete[npi_sys_split_perc_col] = np.nan

                # col req
                the_InitiativePlanningLevel_to_delete = the_InitiativePlanningLevel_to_delete[
                    cols_required_in_output_initiative_pl_level
                ]

                InitiativePlanningLevelDeleted_list.append(the_InitiativePlanningLevel_to_delete)

                # ------------------- 4. Deleting InitiativeLevelAssortment data
                key = [
                    version_col,
                    initiative_col,
                    data_object_col,
                    item_col,
                    planning_account_col,
                    planning_channel_col,
                    planning_region_col,
                    planning_pnl_col,
                    planning_demand_domain_col,
                    loc_col,
                ]
                the_InitiativeLevelAssortment_to_delete = pd.merge(
                    InitiativeLevelAssortment,
                    the_assortment_w_master[key].drop_duplicates(),
                    how="inner",
                )

                the_InitiativeLevelAssortment_to_delete[npi_assorment_final_by_level_col] = np.nan

                # col req
                the_InitiativeLevelAssortment_to_delete = the_InitiativeLevelAssortment_to_delete[
                    cols_required_in_output_initiative_level_ass
                ]

                InitiativeLevelAssortmentDeleted_list.append(
                    the_InitiativeLevelAssortment_to_delete
                )

                # ------------------- 5. Deleting InitiativeAssortment data
                key = [
                    version_col,
                    item_col,
                    planning_account_col,
                    planning_channel_col,
                    planning_region_col,
                    planning_pnl_col,
                    planning_demand_domain_col,
                    loc_col,
                ]
                the_InitiativeAssortment_to_delete = pd.merge(
                    InitiativeAssortment,
                    the_assortment_w_master[key].drop_duplicates(),
                    how="inner",
                )

                the_InitiativeAssortment_to_delete[npi_ass_final_col] = np.nan

                # col req
                the_InitiativeAssortment_to_delete = the_InitiativeAssortment_to_delete[
                    cols_required_in_output_initiative_ass
                ]

                InitiativeAssortmentDeleted_list.append(the_InitiativeAssortment_to_delete)

                # ------------------- 6. Deleting InitiativePlanningLevelNewProductForecast data
                key = [
                    version_col,
                    initiative_col,
                    data_object_col,
                    pl_item_col,
                    planning_account_col,
                    planning_channel_col,
                    planning_region_col,
                    planning_pnl_col,
                    planning_location_col,
                    planning_demand_domain_col,
                ]

                the_InitiativePlanningLevelNewProductForecast_to_delete = pd.merge(
                    InitiativePlanningLevelNewProductForecast,
                    the_assortment_w_master[key].drop_duplicates(),
                    how="inner",
                )

                the_InitiativePlanningLevelNewProductForecast_to_delete[
                    pl_level_cann_impact_col
                ] = np.nan
                the_InitiativePlanningLevelNewProductForecast_to_delete[pl_level_npi_fcst_col] = (
                    np.nan
                )

                # col req
                the_InitiativePlanningLevelNewProductForecast_to_delete = (
                    the_InitiativePlanningLevelNewProductForecast_to_delete[
                        cols_required_in_output_initiative_pl_level_new_prod_fcst
                    ]
                )

                InitiativePlanningLevelNewProductForecastDeleted_list.append(
                    the_InitiativePlanningLevelNewProductForecast_to_delete
                )

                # ------------------- 7. Deleting InitiativeNewProductForecast data
                key = [
                    version_col,
                    initiative_col,
                    pl_item_col,
                    planning_account_col,
                    planning_channel_col,
                    planning_region_col,
                    planning_pnl_col,
                    planning_demand_domain_col,
                    planning_location_col,
                ]
                the_InitiativeNewProductForecast_to_delete = pd.merge(
                    InitiativeNewProductForecast,
                    the_assortment_w_master[key].drop_duplicates(),
                    how="inner",
                )

                the_InitiativeNewProductForecast_to_delete[pl_level_cann_impact_l1_col] = np.nan
                the_InitiativeNewProductForecast_to_delete[pl_level_npi_fcst_l1_col] = np.nan

                # col req
                the_InitiativeNewProductForecast_to_delete = (
                    the_InitiativeNewProductForecast_to_delete[
                        cols_required_in_output_initiative_new_prod_fcst
                    ]
                )

                InitiativeNewProductForecastDeleted_list.append(
                    the_InitiativeNewProductForecast_to_delete
                )

        # Unpack the output
        NPIFcstPublishedDeleted = concat_to_dataframe(NPIFcstPublishedDeleted_list)
        AssortmentNewDeleted = concat_to_dataframe(AssortmentNewDeleted_list)
        if RemoveAssortmentScope == "Full":
            InitiativePlanningLevelDeleted = concat_to_dataframe(
                InitiativePlanningLevelDeleted_list
            )
            InitiativeLevelAssortmentDeleted = concat_to_dataframe(
                InitiativeLevelAssortmentDeleted_list
            )
            InitiativeAssortmentDeleted = concat_to_dataframe(InitiativeAssortmentDeleted_list)
            InitiativePlanningLevelNewProdFcstDeleted = concat_to_dataframe(
                InitiativePlanningLevelNewProductForecastDeleted_list
            )
            InitiativeNewProductForecastDeleted = concat_to_dataframe(
                InitiativeNewProductForecastDeleted_list
            )

        # Remove duplicates
        NPIFcstPublishedDeleted.drop_duplicates(inplace=True)
        AssortmentNewDeleted.drop_duplicates(inplace=True)
        InitiativePlanningLevelDeleted.drop_duplicates(inplace=True)
        InitiativeLevelAssortmentDeleted.drop_duplicates(inplace=True)
        InitiativeAssortmentDeleted.drop_duplicates(inplace=True)
        InitiativePlanningLevelNewProdFcstDeleted.drop_duplicates(inplace=True)
        InitiativeNewProductForecastDeleted.drop_duplicates(inplace=True)

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        logger.exception(e)
        return (
            NPIFcstPublishedDeleted,
            AssortmentNewDeleted,
            InitiativePlanningLevelDeleted,
            InitiativeLevelAssortmentDeleted,
            InitiativeAssortmentDeleted,
            InitiativePlanningLevelNewProdFcstDeleted,
            InitiativeNewProductForecastDeleted,
        )

    return (
        NPIFcstPublishedDeleted,
        AssortmentNewDeleted,
        InitiativePlanningLevelDeleted,
        InitiativeLevelAssortmentDeleted,
        InitiativeAssortmentDeleted,
        InitiativePlanningLevelNewProdFcstDeleted,
        InitiativeNewProductForecastDeleted,
    )
