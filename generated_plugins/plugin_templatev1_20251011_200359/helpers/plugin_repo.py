import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
import logging

logger = logging.getLogger("o9_logger")

def main(ItemMaster, AssortmentFinal, AssortmentSellOut, Date):
    """
    Main function to implement the transition assortment expansion logic.
    """
    try:
        # --- Step 1: Create an Expanded Assortment for 'AssortmentFinal' ---
        logger.info("Step 1: Creating an Expanded Assortment for 'AssortmentFinal'")
        # 1.1 Create AF_Template by finding template rows
        af_template = pd.merge(
            AssortmentFinal,
            ItemMaster[['Planning Item', 'Transition Item']].drop_duplicates(),
            left_on='Item',
            right_on='Planning Item',
            how='inner'
        )

        # 1.2 Create Expanded_AF by generating all combinations
        expanded_af_raw = pd.merge(
            af_template,
            ItemMaster[['Item', 'Transition Item']],
            on='Transition Item',
            suffixes=('_template', '_new')
        )
        
        expanded_af = pd.DataFrame({
            'Version Name': expanded_af_raw['Version Name'],
            'Item': expanded_af_raw['Item_new'],
            'Planning Account': expanded_af_raw['Planning Account'],
            'Planning Channel': expanded_af_raw['Planning Channel'],
            'Planning Region': expanded_af_raw['Planning Region'],
            'Planning PnL': expanded_af_raw['Planning PnL'],
            'Location': expanded_af_raw['Location'],
        })
        expanded_af['Planning Demand Domain'] = expanded_af['Item'] + '-' + expanded_af['Planning Region']
        
        expanded_af.drop_duplicates(inplace=True)

        # --- Step 2: Generate Output 1 - AssortmentFinal_Output ---
        logger.info("Step 2: Generating 'AssortmentFinal_Output'")
        # 2.1 Identify newly created rows via a left anti-join
        af_join_keys = ['Version Name', 'Item', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning PnL', 'Location']
        merged_af = pd.merge(
            expanded_af,
            AssortmentFinal,
            on=af_join_keys,
            how='left',
            indicator=True
        )
        new_af_rows = merged_af[merged_af['_merge'] == 'left_only'].drop(columns=['_merge'])

        # 2.2 Format the final output table
        AssortmentFinal_Output = new_af_rows[expanded_af.columns].copy()
        AssortmentFinal_Output['Assortment Final'] = 1
        AssortmentFinal_Output['Transition Sell In Assortment'] = 1
        
        # Reorder columns to match schema
        assortmentfinal_output_cols = [
            "Version Name", "Item", "Planning Account", "Planning Channel", "Planning Region", 
            "Planning Demand Domain", "Planning PnL", "Location", "Assortment Final", "Transition Sell In Assortment"
        ]
        AssortmentFinal_Output = AssortmentFinal_Output[assortmentfinal_output_cols]

        # --- Step 3: Create an Expanded Assortment for 'AssortmentSellOut' ---
        logger.info("Step 3: Creating an Expanded Assortment for 'AssortmentSellOut'")
        # 3.1 Create ASO_Template
        aso_template = pd.merge(
            AssortmentSellOut,
            ItemMaster[['Planning Item', 'Transition Item']].drop_duplicates(),
            on='Planning Item',
            how='inner'
        )

        # 3.2 Create Expanded_ASO
        expanded_aso_raw = pd.merge(
            aso_template,
            ItemMaster[['Planning Item', 'Transition Item']],
            on='Transition Item',
            suffixes=('_template', '_new')
        )

        expanded_aso = pd.DataFrame({
            'Version Name': expanded_aso_raw['Version Name'],
            'Planning Item': expanded_aso_raw['Planning Item_new'],
            'Planning Account': expanded_aso_raw['Planning Account'],
            'Planning Channel': expanded_aso_raw['Planning Channel'],
            'Planning Region': expanded_aso_raw['Planning Region'],
            'Planning PnL': expanded_aso_raw['Planning PnL'],
            'Planning Location': expanded_aso_raw['Planning Location']
        })
        expanded_aso['Planning Demand Domain'] = expanded_aso['Planning Item'] + '-' + expanded_aso['Planning Region']
        
        expanded_aso.drop_duplicates(inplace=True)

        # --- Step 4: Generate Output 2 - AssortmentSellOut_Output ---
        logger.info("Step 4: Generating 'AssortmentSellOut_Output'")
        # 4.1 Identify newly created rows via a left anti-join
        aso_join_keys = ['Version Name', 'Planning Item', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning PnL', 'Planning Location']
        merged_aso = pd.merge(
            expanded_aso,
            AssortmentSellOut,
            on=aso_join_keys,
            how='left',
            indicator=True
        )
        new_aso_rows = merged_aso[merged_aso['_merge'] == 'left_only'].drop(columns=['_merge'])

        # 4.2 Format the final output table
        AssortmentSellOut_Output = new_aso_rows[expanded_aso.columns].copy()
        AssortmentSellOut_Output['Mdlz DP Assortment Sell Out'] = 1
        AssortmentSellOut_Output['Transition Sell Out Assortment'] = 1
        
        assortmentsellout_output_cols = [
            "Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", 
            "Planning Demand Domain", "Planning PnL", "Planning Location", "Mdlz DP Assortment Sell Out", "Transition Sell Out Assortment"
        ]
        AssortmentSellOut_Output = AssortmentSellOut_Output[assortmentsellout_output_cols]

        # --- Step 5: Generate Output 3 - TransitionFlag_Output ---
        logger.info("Step 5: Generating 'TransitionFlag_Output'")
        # 5.1 Retrieve Transition Start Date
        v_TransitionStartDate = Date['Transition Start Date'].iloc[0]

        # 5.2 Create a unified list of all transition intersections
        af_dimensions = expanded_af.rename(columns={'Item': 'Planning Item', 'Location': 'Planning Location'})
        af_dimensions = af_dimensions[['Version Name', 'Planning Item', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning Demand Domain', 'Planning PnL', 'Planning Location']]
        
        aso_dimensions = expanded_aso[['Version Name', 'Planning Item', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning Demand Domain', 'Planning PnL', 'Planning Location']]

        # Combine using concat_to_dataframe for library consistency, then drop duplicates
        all_transition_intersections = concat_to_dataframe([af_dimensions, aso_dimensions]).drop_duplicates().reset_index(drop=True)

        # 5.3 Format the final output table
        TransitionFlag_Output = all_transition_intersections
        TransitionFlag_Output['Partial Week'] = v_TransitionStartDate
        TransitionFlag_Output['Transition Flag'] = 1
        
        transitionflag_output_cols = [
            "Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", 
            "Planning Demand Domain", "Planning PnL", "Planning Location", "Partial Week", "Transition Flag"
        ]
        TransitionFlag_Output = TransitionFlag_Output[transitionflag_output_cols]

        return AssortmentFinal_Output, AssortmentSellOut_Output, TransitionFlag_Output
        
    except Exception as e:
        logger.error(f"An error occurred in the main logic: {e}")
        # Return empty dataframes matching the output schema in case of an error
        empty_af = pd.DataFrame(columns=["Version Name", "Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Location", "Assortment Final", "Transition Sell In Assortment"])
        empty_aso = pd.DataFrame(columns=["Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Mdlz DP Assortment Sell Out", "Transition Sell Out Assortment"])
        empty_tf = pd.DataFrame(columns=["Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Partial Week", "Transition Flag"])
        return empty_af, empty_aso, empty_tf