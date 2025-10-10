import logging
import pandas as pd
from o9Reference.common_utils.dataframe_utils import create_cartesian_product
from o9Reference.common_utils.decorators import log_inputs_and_outputs, timed

logger = logging.getLogger("o9_logger")

@log_inputs_and_outputs
@timed
def main(ItemMaster, AssortmentFinal, AssortmentSellOut, Date):
    """
    Main function to implement the transition assortment expansion logic.
    """
    # 1. Identify the 'seed items' from AssortmentFinal
    if 'Item' not in AssortmentFinal.columns or AssortmentFinal.empty:
        logger.warning("AssortmentFinal is empty or missing 'Item' column. Exiting.")
        # Return empty dataframes matching the output schema
        return (
            pd.DataFrame(columns=["Version Name", "Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Location", "Assortment Final", "Transition Sell In Assortments"]),
            pd.DataFrame(columns=["Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Mdlz DP Assortment Sell Out", "Transition Sell Out Assortments"]),
            pd.DataFrame(columns=["Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Partial Week", "Transition Flag"])
        )
    seed_items = AssortmentFinal['Item'].unique()

    # 2. Find the 'Transition Item' (base code) for the seed items
    transition_items_df = ItemMaster[ItemMaster['Item'].isin(seed_items)]
    transition_items = transition_items_df['Transition Item'].unique()

    # 3. Get the complete list of all related Item and Planning Item values
    full_item_list_df = ItemMaster[ItemMaster['Transition Item'].isin(transition_items)]
    all_items = full_item_list_df['Item'].unique()
    all_planning_items = full_item_list_df['Planning Item'].unique()

    # 4. Create ExpandedAssortmentFinal
    assort_final_keys = ['Version Name', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning PnL', 'Location']
    unique_assort_final_combos = AssortmentFinal[assort_final_keys].drop_duplicates()
    items_df = pd.DataFrame({'Item': all_items})
    ExpandedAssortmentFinal = create_cartesian_product(unique_assort_final_combos, items_df)
    ExpandedAssortmentFinal['Planning Demand Domain'] = ExpandedAssortmentFinal['Item'].astype(str) + '_' + ExpandedAssortmentFinal['Planning Region'].astype(str)

    # 5. Create ExpandedAssortmentSellOut
    assort_sellout_keys = ['Version Name', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning PnL', 'Planning Location']
    unique_assort_sellout_combos = AssortmentSellOut[assort_sellout_keys].drop_duplicates()
    planning_items_df = pd.DataFrame({'Planning Item': all_planning_items})
    ExpandedAssortmentSellOut = create_cartesian_product(unique_assort_sellout_combos, planning_items_df)
    ExpandedAssortmentSellOut['Planning Demand Domain'] = ExpandedAssortmentSellOut['Planning Item'].astype(str) + '_' + ExpandedAssortmentSellOut['Planning Region'].astype(str)

    # 6. Generate output AssortmentFinal
    key_cols_final = ['Version Name', 'Item', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning PnL', 'Location']
    merged_final = ExpandedAssortmentFinal.merge(
        AssortmentFinal[key_cols_final].drop_duplicates().assign(exists=1),
        on=key_cols_final,
        how='left'
    )
    new_rows_final = merged_final[merged_final['exists'].isna()].copy()
    new_rows_final['Assortment Final'] = 1
    new_rows_final['Transition Sell In Assortments'] = 1
    output_schema_final = ["Version Name", "Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Location", "Assortment Final", "Transition Sell In Assortments"]
    AssortmentFinal_output = new_rows_final[output_schema_final]

    # 7. Generate output AssortmentSellOut
    key_cols_sellout = ['Version Name', 'Planning Item', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning PnL', 'Planning Location']
    merged_sellout = ExpandedAssortmentSellOut.merge(
        AssortmentSellOut[key_cols_sellout].drop_duplicates().assign(exists=1),
        on=key_cols_sellout,
        how='left'
    )
    new_rows_sellout = merged_sellout[merged_sellout['exists'].isna()].copy()
    new_rows_sellout['Mdlz DP Assortment Sell Out'] = 1
    new_rows_sellout['Transition Sell Out Assortments'] = 1
    output_schema_sellout = ["Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Mdlz DP Assortment Sell Out", "Transition Sell Out Assortments"]
    AssortmentSellOut_output = new_rows_sellout[output_schema_sellout]

    # 8. Generate output TransitionFlag
    # a. Combine planning-level rows
    # Align ExpandedAssortmentFinal columns to match ExpandedAssortmentSellOut
    item_map = ItemMaster[['Item', 'Planning Item']].drop_duplicates()
    loc_map = AssortmentSellOut[['Location', 'Planning Location']].drop_duplicates().dropna()
    
    df_final_aligned = ExpandedAssortmentFinal.merge(item_map, on='Item', how='inner')
    if not loc_map.empty:
        df_final_aligned = df_final_aligned.merge(loc_map, on='Location', how='inner')
    else: # Handle case where no location mapping exists
        df_final_aligned['Planning Location'] = None

    # Recalculate PDD based on Planning Item
    df_final_aligned['Planning Demand Domain'] = df_final_aligned['Planning Item'].astype(str) + '_' + df_final_aligned['Planning Region'].astype(str)

    common_cols = ['Version Name', 'Planning Item', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning Demand Domain', 'Planning PnL', 'Planning Location']
    
    combined_df = pd.concat([
        df_final_aligned[common_cols],
        ExpandedAssortmentSellOut[common_cols]
    ]).drop_duplicates().reset_index(drop=True)

    # b. Retrieve and populate Transition Start Date
    if not Date.empty:
        combined_df = combined_df.merge(Date[['Version Name', 'Transition Start Date']], on='Version Name', how='left')
        combined_df.rename(columns={'Transition Start Date': 'Partial Week'}, inplace=True)
    else:
        combined_df['Partial Week'] = pd.NaT

    # c. Add Transition Flag
    combined_df['Transition Flag'] = 1

    output_schema_flag = ["Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Partial Week", "Transition Flag"]
    TransitionFlag_output = combined_df[output_schema_flag]

    return AssortmentFinal_output, AssortmentSellOut_output, TransitionFlag_output