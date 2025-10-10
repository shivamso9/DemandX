import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.common_utils import split_string
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

col_mapping = {"SKU DC Split": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(Actuals, ConsensusFcst, ItemAttribute, SKUDCSplit, df_keys):
    plugin_name = "DP034DCSplitRatio"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    # Configurables
    location_col = "Location.[Location]"
    planning_item_col = "Item.[Planning Item]"
    item_col = "Item.[Item]"
    version_col = "Version.[Version Name]"
    history_measure = "Sell In Stat L0"
    consensus_fcst_col = "Consensus Fcst"
    partial_week_col = "Time.[Partial Week]"
    npi_ml_attribute_col = "Collab Attribute Item"
    npi_ml_attribute_delimiter = ","
    agg_col = "Aggregate"
    indicator_column = "_merge"
    planning_channel_col = "Channel.[Planning Channel]"
    planning_account_col = "Account.[Planning Account]"
    planning_pnl_col = "PnL.[Planning PnL]"
    planning_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    planning_region_col = "Region.[Planning Region]"

    # output measures
    dc_split_ratio_col = "SKU DC Split"
    cols_required_in_output = [
        version_col,
        location_col,
        planning_channel_col,
        planning_account_col,
        planning_pnl_col,
        planning_demand_domain_col,
        planning_region_col,
        item_col,
        partial_week_col,
        dc_split_ratio_col,
    ]
    Output = pd.DataFrame(columns=cols_required_in_output)
    try:
        # Filter required cols
        req_cols = [
            version_col,
            location_col,
            planning_channel_col,
            planning_account_col,
            planning_pnl_col,
            planning_demand_domain_col,
            planning_region_col,
            partial_week_col,
            planning_item_col,
            item_col,
            history_measure,
        ]
        if Actuals is not None:
            Actuals = Actuals[req_cols]
        else:
            logger.info("No actuals present for slice {} ...".format(df_keys))
            logger.warning("Returning empty dataframe")
            return Output

        req_cols = [
            version_col,
            planning_channel_col,
            planning_account_col,
            planning_pnl_col,
            planning_demand_domain_col,
            planning_region_col,
            partial_week_col,
            planning_item_col,
            consensus_fcst_col,
        ]
        ConsensusFcst = ConsensusFcst[req_cols]

        req_cols = [version_col, item_col, npi_ml_attribute_col]
        ItemAttribute = ItemAttribute[req_cols]

        logger.info("Creating Item dataframe ...")
        item_cols = [
            "Item.[Item]",
            "Item.[Item Intro Date]",
            "Item.[Item Disc Date]",
            "Item.[Item Status]",
            "Item.[Is New Item]",
            "Item.[Planning Item]",
            "Item.[Transition Item]",
            "Item.[Stat Item]",
            "Item.[L1]",
            "Item.[L2]",
            "Item.[L3]",
            "Item.[L4]",
            "Item.[L5]",
            "Item.[L6]",
            "Item.[Item Stage]",
            "Item.[Item Type]",
            "Item.[PLC Status]",
            "Item.[Item Class]",
            "Item.[Transition Group]",
            "Item.[A1]",
            "Item.[A2]",
            "Item.[A3]",
            "Item.[A4]",
            "Item.[A5]",
            "Item.[A6]",
            "Item.[A7]",
            "Item.[A8]",
            "Item.[A9]",
            "Item.[A10]",
            "Item.[A11]",
            "Item.[A12]",
            "Item.[A13]",
            "Item.[A14]",
            "Item.[A15]",
            "Item.[All Item]",
        ]
        ItemAttribute = split_string(
            values=list(ItemAttribute[npi_ml_attribute_col]),
            delimiter=npi_ml_attribute_delimiter,
            col_names=item_cols,
        )

        # define item hierarchy
        search_level = [
            "Item.[L1]",
            "Item.[L2]",
            "Item.[L3]",
            "Item.[L4]",
            "Item.[L5]",
            "Item.[L6]",
            "Item.[All Item]",
        ]
        logger.info("search_level : {}".format(search_level))

        if len(Actuals) == 0:
            logger.warning("Actuals is empty for slice : {}".format(df_keys))
            logger.warning("Returning empty dataframe ...")
            return Output

        if len(ConsensusFcst) == 0:
            logger.warning("ConsensusFcst is empty for slice : {}".format(df_keys))
            logger.warning("Returning empty dataframe ...")
            return Output

        # Filter relevant columns from Item Attribute
        req_cols = [
            item_col,
            planning_item_col,
            "Item.[Transition Item]",
            "Item.[L1]",
            "Item.[L2]",
            "Item.[L3]",
            "Item.[L4]",
            "Item.[L5]",
            "Item.[L6]",
            "Item.[All Item]",
        ]
        ItemAttribute = ItemAttribute[req_cols]

        # Join actuals with Item attribute on item col
        Actuals = Actuals.merge(
            ItemAttribute.drop(planning_item_col, axis=1),
            how="inner",
            on=item_col,
        )

        # collect future customer group/pl item/week combinations
        consensus_combinations = ConsensusFcst[
            [
                planning_channel_col,
                planning_account_col,
                planning_pnl_col,
                planning_demand_domain_col,
                planning_region_col,
                planning_item_col,
                partial_week_col,
            ]
        ].drop_duplicates()

        # join with item table and get it down to Item Level
        future_combinations = consensus_combinations.merge(
            ItemAttribute[[item_col, planning_item_col]],
            on=planning_item_col,
            how="inner",
        )

        # drop duplicates after join - one pl item can have many items
        future_combinations = future_combinations[
            [
                planning_channel_col,
                planning_account_col,
                planning_pnl_col,
                planning_demand_domain_col,
                planning_region_col,
                item_col,
                partial_week_col,
            ]
        ].drop_duplicates()

        # identify combinations with consensus fcst but no actuals
        dimensions = [
            planning_channel_col,
            planning_account_col,
            planning_pnl_col,
            planning_demand_domain_col,
            planning_region_col,
            planning_item_col,
        ]
        intersections_with_consensus_fcst = ConsensusFcst[dimensions].drop_duplicates()

        # identify combinations for which sku dc split exist
        intersections_with_sku_dc_split = pd.DataFrame(columns=dimensions)
        if SKUDCSplit is not None:
            intersections_with_sku_dc_split = SKUDCSplit[dimensions].drop_duplicates()

        # perform a left join, with indicator column (says left_only, both and right_only)
        merged_df = intersections_with_consensus_fcst.merge(
            intersections_with_sku_dc_split, how="left", indicator=True
        )

        # collect the intersections which are present in consensus but not in sku dc split
        intersections_with_no_sku_dc_split = merged_df[merged_df[indicator_column] == "left_only"]
        intersections_with_no_sku_dc_split.drop(indicator_column, axis=1, inplace=True)
        logger.info(
            "intersections_with_no_sku_dc_split, shape : {}".format(
                intersections_with_no_sku_dc_split.shape
            )
        )

        # re order columns
        column_order = [
            planning_channel_col,
            planning_account_col,
            planning_pnl_col,
            planning_demand_domain_col,
            planning_region_col,
            planning_item_col,
        ]
        intersections_with_no_sku_dc_split = intersections_with_no_sku_dc_split[column_order]

        req_cols = [
            planning_channel_col,
            planning_account_col,
            planning_pnl_col,
            planning_demand_domain_col,
            planning_region_col,
            location_col,
            item_col,
            dc_split_ratio_col,
        ]

        if len(intersections_with_no_sku_dc_split) == 0:
            logger.warning("intersections_with_no_sku_dc_split is empty ... ")

        all_ratios = []
        logger.info("Looping through all combinations ...")
        for (
            the_planning_channel_col,
            the_planning_account_col,
            the_planning_pnl_col,
            the_planning_demand_domain_col,
            the_planning_region_col,
            the_pl_item,
        ) in intersections_with_no_sku_dc_split.itertuples(index=False):
            logger.info("---- the_planning_channel_col : {}".format(the_planning_channel_col))
            logger.info("---- the_planning_account_col : {}".format(the_planning_account_col))
            logger.info("---- the_planning_pnl_col : {}".format(the_planning_pnl_col))
            logger.info(
                "---- the_planning_demand_domain_col : {}".format(the_planning_demand_domain_col)
            )
            logger.info("---- the_planning_region_col : {}".format(the_planning_region_col))
            logger.info("---- the_pl_item : {}".format(the_pl_item))
            try:
                # collect all attributes for the pl item
                the_pl_item_attributes = ItemAttribute[
                    ItemAttribute[planning_item_col] == the_pl_item
                ]

                if the_pl_item_attributes.empty:
                    logger.warning(
                        "ItemAttributes not found for planning item : {}".format(the_pl_item)
                    )
                    continue

                # get item hierarchy members
                logger.info("searching through item hierarchy to find actuals ...")
                for the_level in search_level:
                    the_level_value = the_pl_item_attributes[the_level].unique()[0]

                    logger.info("--------- {} : {}".format(the_level, the_level_value))

                    # check if actuals are present at the level
                    filter_clause = (
                        (Actuals[planning_channel_col] == the_planning_channel_col)
                        & (Actuals[planning_account_col] == the_planning_account_col)
                        & (Actuals[planning_pnl_col] == the_planning_pnl_col)
                        & (Actuals[planning_demand_domain_col] == the_planning_demand_domain_col)
                        & (Actuals[planning_region_col] == the_planning_region_col)
                        & (Actuals[the_level] == the_level_value)
                    )
                    req_data = Actuals[filter_clause]

                    if len(req_data) > 0:
                        logger.info(
                            "--------- Actuals available at {} level, shape : {} ..".format(
                                the_level, req_data.shape
                            )
                        )
                        fields_to_group = [
                            planning_channel_col,
                            planning_account_col,
                            planning_pnl_col,
                            planning_demand_domain_col,
                            planning_region_col,
                            location_col,
                            the_level,
                        ]
                        the_level_aggregates = (
                            req_data.groupby(fields_to_group).sum()[[history_measure]].reset_index()
                        )

                        # calculate sum of actuals dropping location grain
                        agg_grouping = [
                            planning_channel_col,
                            planning_account_col,
                            planning_pnl_col,
                            planning_demand_domain_col,
                            planning_region_col,
                            the_level,
                        ]
                        the_level_aggregates[agg_col] = the_level_aggregates.groupby(agg_grouping)[
                            history_measure
                        ].transform(sum)

                        # divide actuals by weekly agg to get ratios
                        the_level_aggregates[dc_split_ratio_col] = the_level_aggregates[
                            history_measure
                        ].div(the_level_aggregates[agg_col])

                        # populate zero in ratio where denominator is zero
                        the_level_aggregates.loc[
                            ~np.isfinite(the_level_aggregates[dc_split_ratio_col]),
                            dc_split_ratio_col,
                        ] = 0

                        # Add planning item column for ease of join
                        the_level_aggregates[planning_item_col] = the_pl_item

                        # Join with ItemAttribute to populate all items under the planning item
                        the_level_aggregates = the_level_aggregates.merge(
                            ItemAttribute[[item_col, planning_item_col]],
                            on=planning_item_col,
                            how="inner",
                        )

                        # filter relevant columns
                        the_ratio = the_level_aggregates[req_cols].drop_duplicates()
                        all_ratios.append(the_ratio)

                        # end the search and proceed to next pl item/cust group combination
                        break
                    else:
                        # continue to search the next level in item hierarchy
                        continue

            except Exception as e:
                logger.exception(e)

            logger.info("---- End of Iteration ----")

        # concat all results to dataframe
        all_ratio_df = concat_to_dataframe(all_ratios)

        if len(all_ratio_df) == 0:
            logger.warning("all ratios dataframe is empty ...")
        else:
            logger.info("future_combinations shape : {}".format(future_combinations.shape))

            # join with future combinations in consensus fcst
            Output = future_combinations.merge(
                all_ratio_df,
                on=[
                    planning_channel_col,
                    planning_account_col,
                    planning_pnl_col,
                    planning_demand_domain_col,
                    planning_region_col,
                    item_col,
                ],
                how="inner",
            )

            # add version grain
            input_version = Actuals[version_col].unique()[0]
            Output.insert(0, version_col, input_version)

        logger.info("-------- Output : head ----------")
        logger.info(Output.head())
        logger.info("Output shape : {}".format(Output.shape))

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        Output = pd.DataFrame(columns=cols_required_in_output)

    return Output
