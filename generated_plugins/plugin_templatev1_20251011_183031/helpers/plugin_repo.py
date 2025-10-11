import pandas as pd
import logging

logger = logging.getLogger("o9_logger")

def main(ItemMaster, AssortmentFinal, AssortmentSellOut, Date):
    """
    Generates new assortment rows based on item transitions.

    Args:
        ItemMaster (pd.DataFrame): DataFrame with item hierarchy.
        AssortmentFinal (pd.DataFrame): DataFrame with final assortment data.
        AssortmentSellOut (pd.DataFrame): DataFrame with sell-out assortment data.
        Date (pd.DataFrame): DataFrame containing the transition start date.

    Returns:
        tuple: A tuple of three DataFrames:
               - Output_AssortmentFinal
               - Output_AssortmentSellOut
               - Output_TransitionFlag
    """

    try:
        # --- Column Name Mapping and for new columns ---
        # Input columns
        COL_ITEM = 'Item'
        COL_TRANSITION_ITEM = 'Transition Item'
        COL_PLANNING_ITEM = 'Planning Item'
        COL_VERSION_NAME = 'Version Name'
        COL_PLANNING_ACCOUNT = 'Planning Account'
        COL_PLANNING_CHANNEL = 'Planning Channel'
        COL_PLANNING_REGION = 'Planning Region'
        COL_PLANNING_PNL = 'Planning PnL'
        COL_LOCATION = 'Location'
        COL_PLANNING_LOCATION = 'Planning Location'
        COL_TRANSITION_START_DATE = 'Transition Start Date'
        COL_PLANNING_DEMAND_DOMAIN = 'Planning Demand Domain'
        
        # Output columns
        OUT_COL_ASSORTMENT_FINAL = 'Assortment Final'
        OUT_COL_TRANSITION_SELL_IN = 'Transition Sell In Assortment'
        OUT_COL_MDLZ_DP_ASSORTMENT = 'Mdlz DP Assortment Sell Out'
        OUT_COL_TRANSITION_SELL_OUT = 'Transition Sell Out Assortment'
        OUT_COL_PARTIAL_WEEK = 'Partial Week'
        OUT_COL_TRANSITION_FLAG = 'Transition Flag'

        # 1. Initialize
        logger.info("Step 1: Initializing Transition Start Date.")
        if Date.empty or COL_TRANSITION_START_DATE not in Date.columns:
            raise ValueError("Input 'Date' is empty or missing 'Transition Start Date' column.")
        v_TransitionDate = Date[COL_TRANSITION_START_DATE].iloc[0]

        # Clean ItemMaster by dropping rows where essential columns are null
        ItemMaster = ItemMaster.dropna(subset=[COL_ITEM, COL_TRANSITION_ITEM]).copy()
        
        # 2. & 3. Create Item Hierarchy & Item-to-Basecode Maps (as a DataFrame)
        logger.info("Step 2 & 3: Creating item hierarchy maps.")
        item_to_basecode_map_df = ItemMaster[[COL_ITEM, COL_TRANSITION_ITEM]].drop_duplicates()

        # 4. Generate AssortmentFinal_Expanded
        logger.info("Step 4: Generating AssortmentFinal_Expanded.")
        # Merge to get the Transition Item for each row in AssortmentFinal
        af_with_ti = pd.merge(AssortmentFinal, item_to_basecode_map_df, on=COL_ITEM, how='left')
        af_with_ti = af_with_ti.dropna(subset=[COL_TRANSITION_ITEM])

        # Expand by merging with all items belonging to the same Transition Item
        AssortmentFinal_Expanded = pd.merge(
            af_with_ti,
            item_to_basecode_map_df.rename(columns={COL_ITEM: 'Expanded_Item'}),
            on=COL_TRANSITION_ITEM,
            how='left'
        )

        # Finalize the expanded DataFrame
        AssortmentFinal_Expanded[COL_ITEM] = AssortmentFinal_Expanded['Expanded_Item']
        AssortmentFinal_Expanded[COL_PLANNING_DEMAND_DOMAIN] = AssortmentFinal_Expanded[COL_ITEM].astype(str) + '-' + AssortmentFinal_Expanded[COL_PLANNING_REGION].astype(str)
        
        final_expanded_cols = [
            COL_VERSION_NAME, COL_ITEM, COL_PLANNING_ACCOUNT, COL_PLANNING_CHANNEL,
            COL_PLANNING_REGION, COL_PLANNING_DEMAND_DOMAIN, COL_PLANNING_PNL, COL_LOCATION
        ]
        AssortmentFinal_Expanded = AssortmentFinal_Expanded[final_expanded_cols].drop_duplicates()

        # 5. Generate AssortmentSellOut_Expanded
        logger.info("Step 5: Generating AssortmentSellOut_Expanded.")
        item_to_basecode_map_renamed = item_to_basecode_map_df.rename(columns={COL_ITEM: COL_PLANNING_ITEM})
        
        aso_with_ti = pd.merge(AssortmentSellOut, item_to_basecode_map_renamed, on=COL_PLANNING_ITEM, how='left')
        aso_with_ti = aso_with_ti.dropna(subset=[COL_TRANSITION_ITEM])
        
        AssortmentSellOut_Expanded = pd.merge(
            aso_with_ti,
            item_to_basecode_map_renamed.rename(columns={COL_PLANNING_ITEM: 'Expanded_Planning_Item'}),
            on=COL_TRANSITION_ITEM,
            how='left'
        )

        AssortmentSellOut_Expanded[COL_PLANNING_ITEM] = AssortmentSellOut_Expanded['Expanded_Planning_Item']
        AssortmentSellOut_Expanded[COL_PLANNING_DEMAND_DOMAIN] = AssortmentSellOut_Expanded[COL_PLANNING_ITEM].astype(str) + '-' + AssortmentSellOut_Expanded[COL_PLANNING_REGION].astype(str)

        sellout_expanded_cols = [
            COL_VERSION_NAME, COL_PLANNING_ITEM, COL_PLANNING_ACCOUNT, COL_PLANNING_CHANNEL,
            COL_PLANNING_REGION, COL_PLANNING_DEMAND_DOMAIN, COL_PLANNING_PNL, COL_PLANNING_LOCATION
        ]
        AssortmentSellOut_Expanded = AssortmentSellOut_Expanded[sellout_expanded_cols].drop_duplicates()

        # 6. Generate Output_AssortmentFinal (Anti-Join)
        logger.info("Step 6: Generating Output_AssortmentFinal.")
        af_join_keys = [
            COL_VERSION_NAME, COL_ITEM, COL_PLANNING_ACCOUNT, COL_PLANNING_CHANNEL,
            COL_PLANNING_REGION, COL_PLANNING_PNL, COL_LOCATION
        ]
        
        # Perform an anti-join using a left merge with an indicator.
        merged_af = pd.merge(
            AssortmentFinal_Expanded,
            AssortmentFinal[af_join_keys],
            on=af_join_keys,
            how='left',
            indicator=True
        )
        Output_AssortmentFinal = merged_af[merged_af['_merge'] == 'left_only'].copy()
        
        # Add new columns and select final schema
        Output_AssortmentFinal[OUT_COL_ASSORTMENT_FINAL] = 1
        Output_AssortmentFinal[OUT_COL_TRANSITION_SELL_IN] = 1
        output_af_cols = final_expanded_cols + [OUT_COL_ASSORTMENT_FINAL, OUT_COL_TRANSITION_SELL_IN]
        Output_AssortmentFinal = Output_AssortmentFinal[output_af_cols]

        # 7. Generate Output_AssortmentSellOut (Anti-Join)
        logger.info("Step 7: Generating Output_AssortmentSellOut.")
        aso_join_keys = [
            COL_VERSION_NAME, COL_PLANNING_ITEM, COL_PLANNING_ACCOUNT, COL_PLANNING_CHANNEL,
            COL_PLANNING_REGION, COL_PLANNING_PNL, COL_PLANNING_LOCATION
        ]

        merged_aso = pd.merge(
            AssortmentSellOut_Expanded,
            AssortmentSellOut[aso_join_keys],
            on=aso_join_keys,
            how='left',
            indicator=True
        )
        Output_AssortmentSellOut = merged_aso[merged_aso['_merge'] == 'left_only'].copy()
        
        Output_AssortmentSellOut[OUT_COL_MDLZ_DP_ASSORTMENT] = 1
        Output_AssortmentSellOut[OUT_COL_TRANSITION_SELL_OUT] = 1
        output_aso_cols = sellout_expanded_cols + [OUT_COL_MDLZ_DP_ASSORTMENT, OUT_COL_TRANSITION_SELL_OUT]
        Output_AssortmentSellOut = Output_AssortmentSellOut[output_aso_cols]

        # 8. Generate Output_TransitionFlag
        logger.info("Step 8: Generating Output_TransitionFlag.")
        union_cols = [
            COL_VERSION_NAME, COL_PLANNING_ITEM, COL_PLANNING_ACCOUNT, COL_PLANNING_CHANNEL,
            COL_PLANNING_REGION, COL_PLANNING_DEMAND_DOMAIN, COL_PLANNING_PNL, COL_PLANNING_LOCATION
        ]

        af_for_union = AssortmentFinal_Expanded.rename(columns={COL_ITEM: COL_PLANNING_ITEM, COL_LOCATION: COL_PLANNING_LOCATION})
        af_for_union = af_for_union[union_cols]

        aso_for_union = AssortmentSellOut_Expanded[union_cols]
        
        TransitionFlag_Union = pd.concat([af_for_union, aso_for_union]).drop_duplicates(ignore_index=True)

        TransitionFlag_Union[OUT_COL_PARTIAL_WEEK] = v_TransitionDate
        TransitionFlag_Union[OUT_COL_TRANSITION_FLAG] = 1
        
        Output_TransitionFlag = TransitionFlag_Union.copy()
        
        logger.info("All steps completed successfully.")
        return Output_AssortmentFinal, Output_AssortmentSellOut, Output_TransitionFlag

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}", exc_info=True)
        # Return empty dataframes with correct schema in case of error
        schema_af = [
            'Version Name', 'Item', 'Planning Account', 'Planning Channel', 'Planning Region',
            'Planning Demand Domain', 'Planning PnL', 'Location', 'Assortment Final',
            'Transition Sell In Assortment'
        ]
        schema_aso = [
            'Version Name', 'Planning Item', 'Planning Account', 'Planning Channel',
            'Planning Region', 'Planning Demand Domain', 'Planning PnL', 'Planning Location',
            'Mdlz DP Assortment Sell Out', 'Transition Sell Out Assortment'
        ]
        schema_tf = [
            'Version Name', 'Planning Item', 'Planning Account', 'Planning Channel',
            'Planning Region', 'Planning Demand Domain', 'Planning PnL', 'Planning Location',
            'Partial Week', 'Transition Flag'
        ]
        return (
            pd.DataFrame(columns=schema_af),
            pd.DataFrame(columns=schema_aso),
            pd.DataFrame(columns=schema_tf)
        )