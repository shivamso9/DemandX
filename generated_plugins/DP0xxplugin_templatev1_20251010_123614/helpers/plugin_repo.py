Here is the complete, corrected code for the plugin module.python
import logging
import pandas as pd
from o9.reference.common.o9_decorators import o9_cor_function_for_plugins, o9_timed
from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")
C = o9Constants()


@o9_cor_function_for_plugins
@o9_timed
def main(ItemMaster, AssortmentFinal, AssortmentSellOut, Date, df_keys=None):
    """
    Main function to generate transition assortments based on business logic.
    """
    if df_keys is None:
        df_keys = {}
    plugin_name = "DPXXXTransitionAssortment"
    logger.info("Executing %s for slice %s ...", plugin_name, df_keys)

    # Define output schemas to handle empty inputs gracefully
    schema_assortment_final = {
        C.ITEM: "object",
        C.PLANNING_ACCOUNT: "object",
        C.PLANNING_CHANNEL: "object",
        C.PLANNING_REGION: "object",
        C.PLANNING_DEMAND_DOMAIN: "object",
        C.PLANNING_PNL: "object",
        C.LOCATION: "object",
        "Assortment Final": "int64",
        "Transition Sell In Assortments": "int64",
    }
    schema_assortment_sellout = {
        C.PLANNING_ITEM: "object",
        C.PLANNING_ACCOUNT: "object",
        C.PLANNING_CHANNEL: "object",
        C.PLANNING_REGION: "object",
        C.PLANNING_DEMAND_DOMAIN: "object",
        C.PLANNING_PNL: "object",
        C.PLANNING_LOCATION: "object",
        "Mdlz DP Assortment Sell Out": "int64",
        "Transition Sell Out Assortments": "int64",
    }
    schema_transition_flag = {
        C.PLANNING_ITEM: "object",
        C.PLANNING_ACCOUNT: "object",
        C.PLANNING_CHANNEL: "object",
        C.PLANNING_REGION: "object",
        C.PLANNING_DEMAND_DOMAIN: "object",
        C.PLANNING_PNL: "object",
        C.PLANNING_LOCATION: "object",
        C.PARTIAL_WEEK: "object",
        "Transition Flag": "int64",
    }

    # Graceful exit if critical inputs are empty
    if ItemMaster.empty or (AssortmentFinal.empty and AssortmentSellOut.empty):
        logger.warning(
            "ItemMaster or both Assortment inputs are empty. Returning empty dataframes."
        )
        return (
            pd.DataFrame(columns=schema_assortment_final.keys()).astype(
                schema_assortment_final
            ),
            pd.DataFrame(columns=schema_assortment_sellout.keys()).astype(
                schema_assortment_sellout
            ),
            pd.DataFrame(columns=schema_transition_flag.keys()).astype(
                schema_transition_flag
            ),
        )

    # Proactively drop duplicates from inputs for consistency and performance
    if not AssortmentFinal.empty:
        AssortmentFinal.drop_duplicates(inplace=True)
    if not AssortmentSellOut.empty:
        AssortmentSellOut.drop_duplicates(inplace=True)

    # 1. Create Expanded Temporary Assortments
    logger.info("Step 1: Creating expanded temporary assortments...")

    # Create base mappings once to avoid redundant work
    item_master_map = ItemMaster[
        [C.ITEM, C.PLANNING_ITEM, C.TRANSITION_ITEM]
    ].drop_duplicates()
    pi_to_ti_map = item_master_map[
        [C.PLANNING_ITEM, C.TRANSITION_ITEM]
    ].drop_duplicates()

    # Expand AssortmentFinal
    temp_assortment_final = pd.DataFrame()
    if not AssortmentFinal.empty:
        af_with_ti = pd.merge(
            AssortmentFinal, pi_to_ti_map, on=C.PLANNING_ITEM, how="left"
        ).dropna(subset=[C.TRANSITION_ITEM])

        temp_assortment_final = pd.merge(
            af_with_ti.drop(columns=[C.ITEM, C.PLANNING_ITEM]),
            item_master_map,
            on=C.TRANSITION_ITEM,
        )

    # Expand AssortmentSellOut
    temp_assortment_sellout = pd.DataFrame()
    if not AssortmentSellOut.empty:
        aso_with_ti = pd.merge(
            AssortmentSellOut, pi_to_ti_map, on=C.PLANNING_ITEM, how="left"
        ).dropna(subset=[C.TRANSITION_ITEM])

        temp_assortment_sellout = pd.merge(
            aso_with_ti.drop(columns=[C.PLANNING_ITEM]),
            pi_to_ti_map,  # This map contains Planning Item and Transition Item
            on=C.TRANSITION_ITEM,
        )

    # 2. Enrich Temporary Assortments
    logger.info("Step 2: Enriching temporary assortments with Planning Demand Domain...")
    if not temp_assortment_final.empty:
        # CORRECTED: Use PLANNING_ITEM for consistent PDD calculation.
        # This fixes the uniqueness issue in TransitionFlag.
        temp_assortment_final[C.PLANNING_DEMAND_DOMAIN] = (
            temp_assortment_final[C.PLANNING_ITEM].astype(str)
            + "-"
            + temp_assortment_final[C.PLANNING_REGION].astype(str)
        )

    if not temp_assortment_sellout.empty:
        temp_assortment_sellout[C.PLANNING_DEMAND_DOMAIN] = (
            temp_assortment_sellout[C.PLANNING_ITEM].astype(str)
            + "-"
            + temp_assortment_sellout[C.PLANNING_REGION].astype(str)
        )

    # 3. Generate AssortmentFinal Output
    logger.info("Step 3: Generating new AssortmentFinal output...")
    output_assortment_final = pd.DataFrame(columns=schema_assortment_final.keys())
    if not temp_assortment_final.empty:
        merge_cols = list(AssortmentFinal.columns)
        new_rows_af = temp_assortment_final.merge(
            AssortmentFinal, on=merge_cols, how="left", indicator=True
        ).query('_merge == "left_only"')

        output_assortment_final = new_rows_af[
            list(schema_assortment_final.keys())[:-2]
        ].copy()
        output_assortment_final["Assortment Final"] = 1
        output_assortment_final["Transition Sell In Assortments"] = 1

    # 4. Generate AssortmentSellOut Output
    logger.info("Step 4: Generating new AssortmentSellOut output...")
    output_assortment_sellout = pd.DataFrame(columns=schema_assortment_sellout.keys())
    if not temp_assortment_sellout.empty:
        merge_cols = list(AssortmentSellOut.columns)
        # CORRECTED: No longer need drop_duplicates() as it's handled at the start
        new_rows_aso = temp_assortment_sellout.merge(
            AssortmentSellOut, on=merge_cols, how="left", indicator=True
        ).query('_merge == "left_only"')

        output_assortment_sellout = new_rows_aso[
            list(schema_assortment_sellout.keys())[:-2]
        ].copy()
        output_assortment_sellout["Mdlz DP Assortment Sell Out"] = 1
        output_assortment_sellout["Transition Sell Out Assortments"] = 1

    # 5. Generate TransitionFlag Output
    logger.info("Step 5: Generating TransitionFlag output...")
    transition_start_date = (
        Date[C.TRANSITION_START_DATE].iloc[0] if not Date.empty else None
    )
    output_transition_flag = pd.DataFrame(columns=schema_transition_flag.keys())

    if transition_start_date:
        planning_keys = [
            C.PLANNING_ITEM,
            C.PLANNING_ACCOUNT,
            C.PLANNING_CHANNEL,
            C.PLANNING_REGION,
            C.PLANNING_DEMAND_DOMAIN,
            C.PLANNING_PNL,
            C.PLANNING_LOCATION,
        ]

        final_planning_recs = pd.DataFrame(columns=planning_keys)
        if not temp_assortment_final.empty:
            final_planning_recs = temp_assortment_final.rename(
                columns={C.LOCATION: C.PLANNING_LOCATION}
            )[planning_keys]

        sellout_planning_recs = pd.DataFrame(columns=planning_keys)
        if not temp_assortment_sellout.empty:
            sellout_planning_recs = temp_assortment_sellout[planning_keys]

        # With the PDD calculation now consistent, drop_duplicates will work correctly
        combined_flags = (
            pd.concat([final_planning_recs, sellout_planning_recs], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if not combined_flags.empty:
            combined_flags[C.PARTIAL_WEEK] = transition_start_date
            combined_flags["Transition Flag"] = 1
            output_transition_flag = combined_flags[list(schema_transition_flag.keys())]

    logger.info("Successfully executed %s.", plugin_name)
    return (
        output_assortment_final.astype(schema_assortment_final),
        output_assortment_sellout.astype(schema_assortment_sellout),
        output_transition_flag.astype(schema_transition_flag),
    )