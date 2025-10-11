### REPO FILE ###
import logging
import pandas as pd

logger = logging.getLogger("o9_logger")


def _safe_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Safely concatenates a list of DataFrames, handling empty lists."""
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def process_version(
    item_master_df: pd.DataFrame,
    assortment_final_df: pd.DataFrame,
    assortment_sell_out_df: pd.DataFrame,
    date_df: pd.DataFrame,
):
    """
    Implements the business logic for a single version.
    """
    # Define column names as constants for clarity and maintainability
    VERSION_NAME = "Version Name"
    ITEM = "Item"
    PLANNING_ITEM = "Planning Item"
    TRANSITION_ITEM = "Transition Item"
    PLANNING_ACCOUNT = "Planning Account"
    PLANNING_CHANNEL = "Planning Channel"
    PLANNING_REGION = "Planning Region"
    PLANNING_PNL = "Planning PnL"
    LOCATION = "Location"
    PLANNING_LOCATION = "Planning Location"
    PLANNING_DEMAND_DOMAIN = "Planning Demand Domain"
    TRANSITION_START_DATE = "Transition Start Date"
    PARTIAL_WEEK = "Partial Week"

    # Step 1: Create an Expanded Assortment for 'AssortmentFinal'
    af_template = pd.merge(
        assortment_final_df,
        item_master_df[[PLANNING_ITEM, TRANSITION_ITEM]],
        left_on=ITEM,
        right_on=PLANNING_ITEM,
        how="inner",
    )
    if af_template.empty:
        logger.warning("No templates found in AssortmentFinal. Skipping AssortmentFinal expansion.")
        expanded_af = pd.DataFrame()
    else:
        expanded_af = pd.merge(
            af_template.drop(columns=[PLANNING_ITEM]), # Drop redundant column before merge
            item_master_df[[ITEM, PLANNING_ITEM, TRANSITION_ITEM]],
            on=TRANSITION_ITEM,
            suffixes=("_template", "_new"),
        )
        expanded_af = expanded_af.assign(
            **{
                PLANNING_DEMAND_DOMAIN: expanded_af["Item_new"]
                + "-"
                + expanded_af[PLANNING_REGION],
            }
        ).rename(columns={"Item_new": ITEM})

        expanded_af = expanded_af[[
            VERSION_NAME,
            ITEM,
            PLANNING_ACCOUNT,
            PLANNING_CHANNEL,
            PLANNING_REGION,
            PLANNING_PNL,
            LOCATION,
            PLANNING_DEMAND_DOMAIN,
        ]]

    # Step 2: Generate Output 1 - AssortmentFinal_Output
    assortment_final_output_schema = [
        VERSION_NAME, ITEM, PLANNING_ACCOUNT, PLANNING_CHANNEL, PLANNING_REGION,
        PLANNING_DEMAND_DOMAIN, PLANNING_PNL, LOCATION, "Assortment Final", "Transition Sell In Assortment"
    ]
    if not expanded_af.empty:
        af_join_keys = [
            VERSION_NAME, ITEM, PLANNING_ACCOUNT, PLANNING_CHANNEL,
            PLANNING_REGION, PLANNING_PNL, LOCATION,
        ]
        # Anti-join to find new rows
        merged_af = expanded_af.merge(
            assortment_final_df[af_join_keys], on=af_join_keys, how="left", indicator=True
        )
        new_af_rows = merged_af[merged_af["_merge"] == "left_only"].drop(columns=["_merge"])
        
        assortment_final_output = new_af_rows.assign(
            **{"Assortment Final": 1, "Transition Sell In Assortment": 1}
        )
        # Enforce final schema
        assortment_final_output = assortment_final_output[assortment_final_output_schema]
    else:
        assortment_final_output = pd.DataFrame(columns=assortment_final_output_schema)

    # Step 3: Create an Expanded Assortment for 'AssortmentSellOut'
    aso_template = pd.merge(
        assortment_sell_out_df,
        item_master_df[[PLANNING_ITEM, TRANSITION_ITEM]],
        on=PLANNING_ITEM,
        how="inner",
    )
    if aso_template.empty:
        logger.warning("No templates found in AssortmentSellOut. Skipping AssortmentSellOut expansion.")
        expanded_aso = pd.DataFrame()
    else:
        expanded_aso = pd.merge(
            aso_template,
            item_master_df[[PLANNING_ITEM, TRANSITION_ITEM]],
            on=TRANSITION_ITEM,
            suffixes=("_template", "_new"),
        )
        expanded_aso = expanded_aso.assign(
            **{
                PLANNING_DEMAND_DOMAIN: expanded_aso["Planning Item_new"]
                + "-"
                + expanded_aso[PLANNING_REGION],
            }
        ).rename(columns={"Planning Item_new": PLANNING_ITEM})

        expanded_aso = expanded_aso[[
            VERSION_NAME,
            PLANNING_ITEM,
            PLANNING_ACCOUNT,
            PLANNING_CHANNEL,
            PLANNING_REGION,
            PLANNING_PNL,
            PLANNING_LOCATION,
            PLANNING_DEMAND_DOMAIN,
        ]]

    # Step 4: Generate Output 2 - AssortmentSellOut_Output
    assortment_sellout_output_schema = [
        VERSION_NAME, PLANNING_ITEM, PLANNING_ACCOUNT, PLANNING_CHANNEL, PLANNING_REGION,
        PLANNING_DEMAND_DOMAIN, PLANNING_PNL, PLANNING_LOCATION, "Mdlz DP Assortment Sell Out",
        "Transition Sell Out Assortment"
    ]
    if not expanded_aso.empty:
        aso_join_keys = [
            VERSION_NAME, PLANNING_ITEM, PLANNING_ACCOUNT, PLANNING_CHANNEL,
            PLANNING_REGION, PLANNING_PNL, PLANNING_LOCATION,
        ]
        # Anti-join to find new rows
        merged_aso = expanded_aso.merge(
            assortment_sell_out_df[aso_join_keys], on=aso_join_keys, how="left", indicator=True
        )
        new_aso_rows = merged_aso[merged_aso["_merge"] == "left_only"].drop(columns=["_merge"])
        
        assortment_sellout_output = new_aso_rows.assign(
            **{
                "Mdlz DP Assortment Sell Out": 1,
                "Transition Sell Out Assortment": 1,
            }
        )
        # Enforce final schema
        assortment_sellout_output = assortment_sellout_output[assortment_sellout_output_schema]
    else:
        assortment_sellout_output = pd.DataFrame(columns=assortment_sellout_output_schema)

    # Step 5: Generate Output 3 - TransitionFlag_Output
    transition_flag_output_schema = [
        VERSION_NAME, PLANNING_ITEM, PLANNING_ACCOUNT, PLANNING_CHANNEL, PLANNING_REGION,
        PLANNING_DEMAND_DOMAIN, PLANNING_PNL, PLANNING_LOCATION, PARTIAL_WEEK, "Transition Flag"
    ]
    if date_df.empty or (expanded_af.empty and expanded_aso.empty):
        logger.warning("Cannot generate TransitionFlag due to missing Date or expanded assortment data.")
        transition_flag_output = pd.DataFrame(columns=transition_flag_output_schema)
    else:
        v_transition_start_date = date_df[TRANSITION_START_DATE].iloc[0]

        af_dimensions = pd.DataFrame()
        if not expanded_af.empty:
            af_dimensions = expanded_af.rename(
                columns={ITEM: PLANNING_ITEM, LOCATION: PLANNING_LOCATION}
            )

        aso_dimensions = pd.DataFrame()
        if not expanded_aso.empty:
            aso_dimensions = expanded_aso.copy()

        all_transition_intersections = (
            _safe_concat([af_dimensions, aso_dimensions])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if all_transition_intersections.empty:
            transition_flag_output = pd.DataFrame(columns=transition_flag_output_schema)
        else:
            transition_flag_output = all_transition_intersections.assign(
                **{PARTIAL_WEEK: v_transition_start_date, "Transition Flag": 1}
            )

    return assortment_final_output, assortment_sellout_output, transition_flag_output


def main(ItemMaster, AssortmentFinal, AssortmentSellOut, Date):
    """
    Main function to drive the transition assortment expansion logic.
    """
    # The input DataFrames are assumed to have the correct column names as per the problem description.
    # No renaming is necessary.

    versions = Date["Version Name"].unique()
    af_outputs, aso_outputs, flag_outputs = [], [], []

    for version in versions:
        logger.info(f"Processing Version: {version}")

        # Filter data for the current version
        item_master_filt = ItemMaster.copy()  # ItemMaster is not version-specific
        assortment_final_filt = AssortmentFinal[AssortmentFinal["Version Name"] == version]
        assortment_sell_out_filt = AssortmentSellOut[AssortmentSellOut["Version Name"] == version]
        date_filt = Date[Date["Version Name"] == version]

        # Check for empty inputs for the version
        if assortment_final_filt.empty and assortment_sell_out_filt.empty:
            logger.warning(f"No assortment data for version {version}. Skipping.")
            continue
        if date_filt.empty:
            logger.warning(f"No date information for version {version}. Skipping.")
            continue

        af_out, aso_out, flag_out = process_version(
            item_master_df=item_master_filt,
            assortment_final_df=assortment_final_filt,
            assortment_sell_out_df=assortment_sell_out_filt,
            date_df=date_filt,
        )
        if not af_out.empty:
            af_outputs.append(af_out)
        if not aso_out.empty:
            aso_outputs.append(aso_out)
        if not flag_out.empty:
            flag_outputs.append(flag_out)

    # Combine results from all versions
    assortment_final_output_final = _safe_concat(af_outputs)
    assortment_sellout_output_final = _safe_concat(aso_outputs)
    transition_flag_output_final = _safe_concat(flag_outputs)
    
    # Return a dictionary of DataFrames as per standard plugin output structure
    return {
        "AssortmentFinal": assortment_final_output_final,
        "AssortmentSellOut": assortment_sellout_output_final,
        "TransitionFlag": transition_flag_output_final,
    }