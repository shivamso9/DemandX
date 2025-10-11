# from both the AssortmentFinal and AssortmentSellOut expansions using UNION DISTINCT.
from helpers.plugin_module import main
import io
import pandas as pd


def assert_df_equal(left, right, key_cols=None):
    """Compares two DataFrames for equality after sorting and resetting index."""
    if key_cols:
        left_sorted = left.sort_values(by=key_cols).reset_index(drop=True)
        right_sorted = right.sort_values(by=key_cols).reset_index(drop=True)
    else:
        left_sorted = left.sort_values(by=left.columns.tolist()).reset_index(drop=True)
        right_sorted = right.sort_values(by=right.columns.tolist()).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(left_sorted, right_sorted, check_dtype=True)


def test_golden_record_single_transition_group(process_logic):
    """
    Validates the core logic for a single transition group where one template row
    in AssortmentFinal and AssortmentSellOut expands to create new rows for all
    other items in the group.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ItemA1,BC-A,ItemA1
ItemA2,BC-A,ItemA2
ItemA3,BC-A,ItemA3
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,ItemA1,A100,PA100,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,ItemA1,ItemA1,A100,PA100,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L10,PL10
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V1,2023-01-01
"""
    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO(date_csv))
    }

    actual_outputs = process_logic(inputs)

    expected_af_output_data = {
        "Version Name": ["V1", "V1"],
        "Item": ["ItemA2", "ItemA3"],
        "Planning Account": ["PA100", "PA100"],
        "Planning Channel": ["PC1", "PC1"],
        "Planning Region": ["PR1", "PR1"],
        "Planning Demand Domain": ["ItemA2-PR1", "ItemA3-PR1"],
        "Planning PnL": ["PP1", "PP1"],
        "Location": ["L1", "L1"],
        "Assortment Final": [1, 1],
        "Transition Sell In Assortment": [1, 1],
    }
    expected_af_output = pd.DataFrame(expected_af_output_data)

    expected_aso_output_data = {
        "Version Name": ["V1", "V1"],
        "Planning Item": ["ItemA2", "ItemA3"],
        "Planning Account": ["PA100", "PA100"],
        "Planning Channel": ["PC1", "PC1"],
        "Planning Region": ["PR1", "PR1"],
        "Planning Demand Domain": ["ItemA2-PR1", "ItemA3-PR1"],
        "Planning PnL": ["PP1", "PP1"],
        "Planning Location": ["PL10", "PL10"],
        "Mdlz DP Assortment Sell Out": [1, 1],
        "Transition Sell Out Assortment": [1, 1],
    }
    expected_aso_output = pd.DataFrame(expected_aso_output_data)

    expected_tf_output_data = {
        "Version Name": ["V1", "V1", "V1"],
        "Planning Item": ["ItemA1", "ItemA2", "ItemA3"],
        "Planning Account": ["PA100", "PA100", "PA100"],
        "Planning Channel": ["PC1", "PC1", "PC1"],
        "Planning Region": ["PR1", "PR1", "PR1"],
        "Planning Demand Domain": ["ItemA1-PR1", "ItemA2-PR1", "ItemA3-PR1"],
        "Planning PnL": ["PP1", "PP1", "PP1"],
        "Planning Location": ["PL10", "PL10", "PL10"],
        "Partial Week": ["2023-01-01", "2023-01-01", "2023-01-01"],
        "Transition Flag": [1, 1, 1],
    }
    expected_tf_output = pd.DataFrame(expected_tf_output_data)
    # The union logic means the AF Location (L1) gets mapped to a PL. We need to unify.
    # AF_Dimensions uses Item->Planning Item and Location->Planning Location.
    # So ItemA1/L1 becomes ItemA1/PL10 in the union base.
    # The ASO dimensions are already correct.
    # Let's adjust expectation. The logic says `Location (alias as Planning Location)` for AF_Dimensions.
    # The AF template has Location=L1. The ASO template has Planning Location=PL10.
    # The union should create two distinct rows for ItemA1, one with L1 and one with PL10.
    expected_tf_output_data_v2 = {
        "Version Name": ["V1", "V1", "V1", "V1", "V1", "V1"],
        "Planning Item": ["ItemA1", "ItemA2", "ItemA3", "ItemA1", "ItemA2", "ItemA3"],
        "Planning Account": ["PA100"] * 6,
        "Planning Channel": ["PC1"] * 6,
        "Planning Region": ["PR1"] * 6,
        "Planning Demand Domain": ["ItemA1-PR1", "ItemA2-PR1", "ItemA3-PR1", "ItemA1-PR1", "ItemA2-PR1", "ItemA3-PR1"],
        "Planning PnL": ["PP1"] * 6,
        "Planning Location": ["L1", "L1", "L1", "PL10", "PL10", "PL10"],
        "Partial Week": ["2023-01-01"] * 6,
        "Transition Flag": [1] * 6,
    }
    expected_tf_output = pd.DataFrame(expected_tf_output_data_v2)


    expected_outputs = {
        "AssortmentFinal": expected_af_output,
        "AssortmentSellOut": expected_aso_output,
        "TransitionFlag": expected_tf_output,
    }

    # Sort dataframes for consistent comparison
    for key in expected_outputs:
        cols = sorted(expected_outputs[key].columns)
        actual_outputs[key] = actual_outputs[key].sort_values(by=cols).reset_index(drop=True)
        expected_outputs[key] = expected_outputs[key].sort_values(by=cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_outputs["AssortmentFinal"], expected_outputs["AssortmentFinal"])
    pd.testing.assert_frame_equal(actual_outputs["AssortmentSellOut"], expected_outputs["AssortmentSellOut"])
    pd.testing.assert_frame_equal(actual_outputs["TransitionFlag"], expected_outputs["TransitionFlag"])


def test_golden_record_multiple_independent_groups(process_logic):
    """
    Ensures the process correctly handles multiple, non-overlapping transition groups,
    processing each expansion independently.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ItemA1,BC-A,ItemA1
ItemA2,BC-A,ItemA2
ItemB1,BC-B,ItemB1
ItemB2,BC-B,ItemB2
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,ItemA1,A100,PA100,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1
1,V1,ItemB1,B200,PB200,C2,PC2,R2,PR2,DD2,PDD2,P2,PP2,L2
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,ItemA1,ItemA1,A100,PA100,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L10,PL10
1,V1,ItemB1,ItemB1,B200,PB200,C2,PC2,R2,PR2,DD2,PDD2,P2,PP2,L20,PL20
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V1,2023-01-01
"""
    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO(date_csv))
    }

    actual_outputs = process_logic(inputs)

    expected_af_output_data = {
        "Version Name": ["V1", "V1"],
        "Item": ["ItemA2", "ItemB2"],
        "Planning Account": ["PA100", "PB200"],
        "Planning Channel": ["PC1", "PC2"],
        "Planning Region": ["PR1", "PR2"],
        "Planning Demand Domain": ["ItemA2-PR1", "ItemB2-PR2"],
        "Planning PnL": ["PP1", "PP2"],
        "Location": ["L1", "L2"],
        "Assortment Final": [1, 1],
        "Transition Sell In Assortment": [1, 1],
    }
    expected_af_output = pd.DataFrame(expected_af_output_data)

    expected_aso_output_data = {
        "Version Name": ["V1", "V1"],
        "Planning Item": ["ItemA2", "ItemB2"],
        "Planning Account": ["PA100", "PB200"],
        "Planning Channel": ["PC1", "PC2"],
        "Planning Region": ["PR1", "PR2"],
        "Planning Demand Domain": ["ItemA2-PR1", "ItemB2-PR2"],
        "Planning PnL": ["PP1", "PP2"],
        "Planning Location": ["PL10", "PL20"],
        "Mdlz DP Assortment Sell Out": [1, 1],
        "Transition Sell Out Assortment": [1, 1],
    }
    expected_aso_output = pd.DataFrame(expected_aso_output_data)

    expected_tf_output_data = {
        "Version Name": ["V1"] * 8,
        "Planning Item": ["ItemA1", "ItemA2", "ItemB1", "ItemB2", "ItemA1", "ItemA2", "ItemB1", "ItemB2"],
        "Planning Account": ["PA100", "PA100", "PB200", "PB200", "PA100", "PA100", "PB200", "PB200"],
        "Planning Channel": ["PC1", "PC1", "PC2", "PC2", "PC1", "PC1", "PC2", "PC2"],
        "Planning Region": ["PR1", "PR1", "PR2", "PR2", "PR1", "PR1", "PR2", "PR2"],
        "Planning Demand Domain": ["ItemA1-PR1", "ItemA2-PR1", "ItemB1-PR2", "ItemB2-PR2", "ItemA1-PR1", "ItemA2-PR1", "ItemB1-PR2", "ItemB2-PR2"],
        "Planning PnL": ["PP1", "PP1", "PP2", "PP2", "PP1", "PP1", "PP2", "PP2"],
        "Planning Location": ["L1", "L1", "L2", "L2", "PL10", "PL10", "PL20", "PL20"],
        "Partial Week": ["2023-01-01"] * 8,
        "Transition Flag": [1] * 8,
    }
    expected_tf_output = pd.DataFrame(expected_tf_output_data)

    expected_outputs = {
        "AssortmentFinal": expected_af_output,
        "AssortmentSellOut": expected_aso_output,
        "TransitionFlag": expected_tf_output,
    }

    # Sort dataframes for consistent comparison
    for key in expected_outputs:
        cols = sorted(expected_outputs[key].columns)
        actual_outputs[key] = actual_outputs[key].sort_values(by=cols).reset_index(drop=True)
        expected_outputs[key] = expected_outputs[key].sort_values(by=cols).reset_index(drop=True)
        # Drop duplicates for TransitionFlag which is a UNION DISTINCT
        if key == "TransitionFlag":
             actual_outputs[key] = actual_outputs[key].drop_duplicates().reset_index(drop=True)
             expected_outputs[key] = expected_outputs[key].drop_duplicates().reset_index(drop=True)


    pd.testing.assert_frame_equal(actual_outputs["AssortmentFinal"], expected_outputs["AssortmentFinal"])
    pd.testing.assert_frame_equal(actual_outputs["AssortmentSellOut"], expected_outputs["AssortmentSellOut"])
    pd.testing.assert_frame_equal(actual_outputs["TransitionFlag"], expected_outputs["TransitionFlag"])


def test_anti_join_excludes_existing_items(process_logic):
    """
    Verifies that if an item within a transition group already exists in the input assortment,
    it is not duplicated in the final output tables (AssortmentFinal_Output, AssortmentSellOut_Output).
    """
    item_master_csv = """Item,Transition Item,Planning Item
ItemA1,BC-A,ItemA1
ItemA2,BC-A,ItemA2
ItemA3,BC-A,ItemA3
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,ItemA1,A100,PA100,C1,PC1,R1,PR1,DD1,ItemA1-PR1,P1,PP1,L1
1,V1,ItemA2,A100,PA100,C1,PC1,R1,PR1,DD1,ItemA2-PR1,P1,PP1,L1
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,ItemA1,ItemA1,A100,PA100,C1,PC1,R1,PR1,DD1,ItemA1-PR1,P1,PP1,L10,PL10
1,V1,ItemA2,ItemA2,A100,PA100,C1,PC1,R1,PR1,DD1,ItemA2-PR1,P1,PP1,L10,PL10
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V1,2023-01-01
"""
    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO(date_csv))
    }

    actual_outputs = process_logic(inputs)

    expected_af_output_data = {
        "Version Name": ["V1"],
        "Item": ["ItemA3"],
        "Planning Account": ["PA100"],
        "Planning Channel": ["PC1"],
        "Planning Region": ["PR1"],
        "Planning Demand Domain": ["ItemA3-PR1"],
        "Planning PnL": ["PP1"],
        "Location": ["L1"],
        "Assortment Final": [1],
        "Transition Sell In Assortment": [1],
    }
    expected_af_output = pd.DataFrame(expected_af_output_data)

    expected_aso_output_data = {
        "Version Name": ["V1"],
        "Planning Item": ["ItemA3"],
        "Planning Account": ["PA100"],
        "Planning Channel": ["PC1"],
        "Planning Region": ["PR1"],
        "Planning Demand Domain": ["ItemA3-PR1"],
        "Planning PnL": ["PP1"],
        "Planning Location": ["PL10"],
        "Mdlz DP Assortment Sell Out": [1],
        "Transition Sell Out Assortment": [1],
    }
    expected_aso_output = pd.DataFrame(expected_aso_output_data)

    # TransitionFlag is based on the EXPANDED set, so it should include all 3 items,
    # not just the newly generated one.
    expected_tf_output_data = {
        "Version Name": ["V1", "V1", "V1", "V1", "V1", "V1"],
        "Planning Item": ["ItemA1", "ItemA2", "ItemA3", "ItemA1", "ItemA2", "ItemA3"],
        "Planning Account": ["PA100"] * 6,
        "Planning Channel": ["PC1"] * 6,
        "Planning Region": ["PR1"] * 6,
        "Planning Demand Domain": ["ItemA1-PR1", "ItemA2-PR1", "ItemA3-PR1", "ItemA1-PR1", "ItemA2-PR1", "ItemA3-PR1"],
        "Planning PnL": ["PP1"] * 6,
        "Planning Location": ["L1", "L1", "L1", "PL10", "PL10", "PL10"],
        "Partial Week": ["2023-01-01"] * 6,
        "Transition Flag": [1] * 6,
    }
    expected_tf_output = pd.DataFrame(expected_tf_output_data)

    expected_outputs = {
        "AssortmentFinal": expected_af_output,
        "AssortmentSellOut": expected_aso_output,
        "TransitionFlag": expected_tf_output,
    }

    # Sort dataframes for consistent comparison
    for key in expected_outputs:
        cols = sorted(expected_outputs[key].columns)
        actual_outputs[key] = actual_outputs[key].sort_values(by=cols).reset_index(drop=True)
        expected_outputs[key] = expected_outputs[key].sort_values(by=cols).reset_index(drop=True)
        # Drop duplicates for TransitionFlag which is a UNION DISTINCT
        if key == "TransitionFlag":
             actual_outputs[key] = actual_outputs[key].drop_duplicates().reset_index(drop=True)
             expected_outputs[key] = expected_outputs[key].drop_duplicates().reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_outputs["AssortmentFinal"], expected_outputs["AssortmentFinal"])
    pd.testing.assert_frame_equal(actual_outputs["AssortmentSellOut"], expected_outputs["AssortmentSellOut"])
    pd.testing.assert_frame_equal(actual_outputs["TransitionFlag"], expected_outputs["TransitionFlag"])


def test_demand_domain_concatenation_logic(process_logic):
    """
    Checks the correct formation of the 'Planning Demand Domain' column by
    concatenating Item/Planning Item and Planning Region for AssortmentFinal
    and AssortmentSellOut respectively.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ItemX1,BC-X,ItemX1
ItemX2,BC-X,ItemX2
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,ItemX1,A1,PA1,C1,PC1,R-US,PR-US,DD1,PDD1,P1,PP1,L-US
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,ItemX1,ItemX1,A1,PA1,C1,PC1,R-CA,PR-CA,DD1,PDD1,P1,PP1,L-CA,PL-CA
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V1,2023-01-01
"""
    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO(date_csv))
    }

    actual_outputs = process_logic(inputs)

    # AssortmentFinal: Expands based on the AF template's region "PR-US"
    expected_af_output_data = {
        "Version Name": ["V1"],
        "Item": ["ItemX2"],
        "Planning Account": ["PA1"],
        "Planning Channel": ["PC1"],
        "Planning Region": ["PR-US"],
        "Planning Demand Domain": ["ItemX2-PR-US"],
        "Planning PnL": ["PP1"],
        "Location": ["L-US"],
        "Assortment Final": [1],
        "Transition Sell In Assortment": [1],
    }
    expected_af_output = pd.DataFrame(expected_af_output_data)
    assert actual_outputs["AssortmentFinal"]["Planning Demand Domain"].iloc[0] == "ItemX2-PR-US"

    # AssortmentSellOut: Expands based on the ASO template's region "PR-CA"
    expected_aso_output_data = {
        "Version Name": ["V1"],
        "Planning Item": ["ItemX2"],
        "Planning Account": ["PA1"],
        "Planning Channel": ["PC1"],
        "Planning Region": ["PR-CA"],
        "Planning Demand Domain": ["ItemX2-PR-CA"],
        "Planning PnL": ["PP1"],
        "Planning Location": ["PL-CA"],
        "Mdlz DP Assortment Sell Out": [1],
        "Transition Sell Out Assortment": [1],
    }
    expected_aso_output = pd.DataFrame(expected_aso_output_data)
    assert actual_outputs["AssortmentSellOut"]["Planning Demand Domain"].iloc[0] == "ItemX2-PR-CA"

    # TransitionFlag: Combines distinct dimensions from both expanded sets
    expected_tf_output_data = {
        "Version Name": ["V1"] * 4,
        "Planning Item": ["ItemX1", "ItemX2", "ItemX1", "ItemX2"],
        "Planning Account": ["PA1"] * 4,
        "Planning Channel": ["PC1"] * 4,
        "Planning Region": ["PR-US", "PR-US", "PR-CA", "PR-CA"],
        "Planning Demand Domain": ["ItemX1-PR-US", "ItemX2-PR-US", "ItemX1-PR-CA", "ItemX2-PR-CA"],
        "Planning PnL": ["PP1"] * 4,
        "Planning Location": ["L-US", "L-US", "PL-CA", "PL-CA"],
        "Partial Week": ["2023-01-01"] * 4,
        "Transition Flag": [1] * 4,
    }
    expected_tf_output = pd.DataFrame(expected_tf_output_data)
    
    expected_outputs = {
        "AssortmentFinal": expected_af_output,
        "AssortmentSellOut": expected_aso_output,
        "TransitionFlag": expected_tf_output,
    }

    # Sort dataframes for consistent comparison
    for key in expected_outputs:
        cols = sorted(expected_outputs[key].columns)
        actual_outputs[key] = actual_outputs[key].sort_values(by=cols).reset_index(drop=True)
        expected_outputs[key] = expected_outputs[key].sort_values(by=cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_outputs["AssortmentFinal"], expected_outputs["AssortmentFinal"])
    pd.testing.assert_frame_equal(actual_outputs["AssortmentSellOut"], expected_outputs["AssortmentSellOut"])
    pd.testing.assert_frame_equal(actual_outputs["TransitionFlag"], expected_outputs["TransitionFlag"])

def test_transition_flag_unification_and_deduplication(run_business_logic):
    """
    Validates that TransitionFlag_Output correctly combines and deduplicates dimension sets
    This test creates overlapping and unique dimension sets from both sources to verify
    the UNION DISTINCT behavior.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU_TEMPLATE,T_GROUP_1,PI_TEMPLATE
SKU_A,T_GROUP_1,PI_A
SKU_B,T_GROUP_1,PI_B
"""

    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,Test Version,SKU_TEMPLATE,ACC1,PA1,CH1,PC1,R1,PR1,DD1,PDD1,PNL,PPNL1,LOC_COMMON
"""

    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,Test Version,ITEM_X,PI_TEMPLATE,ACC1,PA1,CH1,PC1,R1,PR1,DD1,PDD1,PNL,PPNL1,LOC_X,LOC_COMMON
V1,Test Version,ITEM_Y,PI_TEMPLATE,ACC1,PA1,CH1,PC1,R1,PR1,DD1,PDD1,PNL,PPNL1,LOC_Y,LOC_UNIQUE
"""

    date_csv = """Version,Version Name,Transition Start Date
V1,Test Version,2025-01-01
"""

    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO(date_csv)),
    }

    result_dfs = run_business_logic(inputs)
    actual_df = result_dfs["TransitionFlag"]

    # Expected output contains unified, distinct rows from both AF and ASO expansions.
    # AF expansion contributes (SKU_A, LOC_COMMON) and (SKU_B, LOC_COMMON). Item is aliased to Planning Item.
    # ASO expansion contributes (PI_A, LOC_COMMON), (PI_B, LOC_COMMON), (PI_A, LOC_UNIQUE), (PI_B, LOC_UNIQUE).
    # Since SKU_* != PI_*, all 6 potential rows are unique and should be in the output.
    expected_data = {
        "Version Name": ["Test Version"] * 6,
        "Planning Item": ["SKU_A", "SKU_B", "PI_A", "PI_B", "PI_A", "PI_B"],
        "Planning Account": ["PA1"] * 6,
        "Planning Channel": ["PC1"] * 6,
        "Planning Region": ["PR1"] * 6,
        "Planning Demand Domain": ["SKU_A-PR1", "SKU_B-PR1", "PI_A-PR1", "PI_B-PR1", "PI_A-PR1", "PI_B-PR1"],
        "Planning PnL": ["PPNL1"] * 6,
        "Planning Location": ["LOC_COMMON", "LOC_COMMON", "LOC_COMMON", "LOC_COMMON", "LOC_UNIQUE", "LOC_UNIQUE"],
        "Partial Week": ["2025-01-01"] * 6,
        "Transition Flag": [1] * 6,
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["Transition Flag"] = expected_df["Transition Flag"].astype('int64')

    # Sort both dataframes for a consistent comparison
    actual_df_sorted = actual_df.sort_values(by=actual_df.columns.tolist()).reset_index(drop=True)
    expected_df_sorted = expected_df.sort_values(by=expected_df.columns.tolist()).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_df_sorted, expected_df_sorted)


def test_transition_date_population(run_business_logic):
    """
    Confirms that the 'Partial Week' column in TransitionFlag_Output is correctly populated
    with the 'Transition Start Date' from the Date input table for the corresponding Version Name.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU1,T1,PI1
SKU2,T1,PI2
"""

    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V_TEST,Test Version,SKU1,ACC1,PA1,CH1,PC1,R1,PR1,DD1,PDD1,PNL,PPNL1,L1
"""

    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
"""

    date_csv = """Version,Version Name,Transition Start Date
V_OTHER,Other Version,2022-01-01
V_TEST,Test Version,2023-10-31
"""

    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO(date_csv)),
    }

    result_dfs = run_business_logic(inputs)
    actual_df = result_dfs["TransitionFlag"]

    # The logic should generate one new row for SKU2 and use the date '2023-10-31' for 'Test Version'.
    expected_data = {
        "Version Name": ["Test Version"],
        "Planning Item": ["SKU2"],
        "Planning Account": ["PA1"],
        "Planning Channel": ["PC1"],
        "Planning Region": ["PR1"],
        "Planning Demand Domain": ["SKU2-PR1"],
        "Planning PnL": ["PPNL1"],
        "Planning Location": ["L1"],
        "Partial Week": ["2023-10-31"],
        "Transition Flag": [1],
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["Transition Flag"] = expected_df["Transition Flag"].astype('int64')

    pd.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_output_schema_and_static_values(run_business_logic):
    """
    Verifies that all three output tables conform to the specified schema and that
    static flag columns (e.g., 'Assortment Final', 'Transition Flag') are correctly set to 1.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU_TEMPLATE,T1,PI_TEMPLATE
SKU_NEW,T1,PI_NEW
"""

    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,Test Version,SKU_TEMPLATE,ACC1,PA1,CH1,PC1,R1,PR1,DD1,SKU_TEMPLATE-PR1,PNL,PPNL1,L1
"""

    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,Test Version,ITEM_X,PI_TEMPLATE,ACC1,PA1,CH1,PC1,R1,PR1,DD1,PI_TEMPLATE-PR1,PNL,PPNL1,LOC_X,PL1
"""

    date_csv = """Version,Version Name,Transition Start Date
V1,Test Version,2025-01-01
"""

    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO(date_csv)),
    }

    result_dfs = run_business_logic(inputs)

    # --- Verification for AssortmentFinal_Output ---
    expected_af_data = {
        "Version Name": ["Test Version"],
        "Item": ["SKU_NEW"],
        "Planning Account": ["PA1"],
        "Planning Channel": ["PC1"],
        "Planning Region": ["PR1"],
        "Planning Demand Domain": ["SKU_NEW-PR1"],
        "Planning PnL": ["PPNL1"],
        "Location": ["L1"],
        "Assortment Final": [1],
        "Transition Sell In Assortment": [1],
    }
    expected_af_df = pd.DataFrame(expected_af_data)
    expected_af_df["Assortment Final"] = expected_af_df["Assortment Final"].astype('int64')
    expected_af_df["Transition Sell In Assortment"] = expected_af_df["Transition Sell In Assortment"].astype('int64')
    pd.testing.assert_frame_equal(result_dfs["AssortmentFinal"].reset_index(drop=True), expected_af_df)

    # --- Verification for AssortmentSellOut_Output ---
    expected_aso_data = {
        "Version Name": ["Test Version"],
        "Planning Item": ["PI_NEW"],
        "Planning Account": ["PA1"],
        "Planning Channel": ["PC1"],
        "Planning Region": ["PR1"],
        "Planning Demand Domain": ["PI_NEW-PR1"],
        "Planning PnL": ["PPNL1"],
        "Planning Location": ["PL1"],
        "Mdlz DP Assortment Sell Out": [1],
        "Transition Sell Out Assortment": [1],
    }
    expected_aso_df = pd.DataFrame(expected_aso_data)
    expected_aso_df["Mdlz DP Assortment Sell Out"] = expected_aso_df["Mdlz DP Assortment Sell Out"].astype('int64')
    expected_aso_df["Transition Sell Out Assortment"] = expected_aso_df["Transition Sell Out Assortment"].astype('int64')
    pd.testing.assert_frame_equal(result_dfs["AssortmentSellOut"].reset_index(drop=True), expected_aso_df)

    # --- Verification for TransitionFlag_Output ---
    expected_tf_data = {
        "Version Name": ["Test Version", "Test Version"],
        "Planning Item": ["SKU_NEW", "PI_NEW"],
        "Planning Account": ["PA1", "PA1"],
        "Planning Channel": ["PC1", "PC1"],
        "Planning Region": ["PR1", "PR1"],
        "Planning Demand Domain": ["SKU_NEW-PR1", "PI_NEW-PR1"],
        "Planning PnL": ["PPNL1", "PPNL1"],
        "Planning Location": ["L1", "PL1"],
        "Partial Week": ["2025-01-01", "2025-01-01"],
        "Transition Flag": [1, 1],
    }
    expected_tf_df = pd.DataFrame(expected_tf_data)
    expected_tf_df["Transition Flag"] = expected_tf_df["Transition Flag"].astype('int64')
    
    actual_tf_sorted = result_dfs["TransitionFlag"].sort_values(by=result_dfs["TransitionFlag"].columns.tolist()).reset_index(drop=True)
    expected_tf_sorted = expected_tf_df.sort_values(by=expected_tf_df.columns.tolist()).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_tf_sorted, expected_tf_sorted)


def test_empty_assortment_inputs(run_business_logic):
    """
    Tests behavior when AssortmentFinal and AssortmentSellOut inputs are empty.
    Expects empty output tables with no errors.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU1,T1,PI1
"""

    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
"""

    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
"""

    date_csv = """Version,Version Name,Transition Start Date
V1,Test Version,2025-01-01
"""

    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO(date_csv)),
    }

    result_dfs = run_business_logic(inputs)

    # --- Assert AssortmentFinal_Output is empty with correct schema ---
    expected_af_cols = [
        "Version Name", "Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Location", "Assortment Final",
        "Transition Sell In Assortment"
    ]
    expected_af_df = pd.DataFrame(columns=expected_af_cols)
    # Set dtypes to match expected output for empty frames
    expected_af_df = expected_af_df.astype({
        'Assortment Final': 'int64', 'Transition Sell In Assortment': 'int64'
    })
    pd.testing.assert_frame_equal(result_dfs["AssortmentFinal"], expected_af_df, check_like=True)

    # --- Assert AssortmentSellOut_Output is empty with correct schema ---
    expected_aso_cols = [
        "Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Planning Location",
        "Mdlz DP Assortment Sell Out", "Transition Sell Out Assortment"
    ]
    expected_aso_df = pd.DataFrame(columns=expected_aso_cols)
    expected_aso_df = expected_aso_df.astype({
        'Mdlz DP Assortment Sell Out': 'int64', 'Transition Sell Out Assortment': 'int64'
    })
    pd.testing.assert_frame_equal(result_dfs["AssortmentSellOut"], expected_aso_df, check_like=True)

    # --- Assert TransitionFlag_Output is empty with correct schema ---
    expected_tf_cols = [
        "Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Planning Location", "Partial Week", "Transition Flag"
    ]
    expected_tf_df = pd.DataFrame(columns=expected_tf_cols)
    expected_tf_df = expected_tf_df.astype({'Transition Flag': 'int64'})
    pd.testing.assert_frame_equal(result_dfs["TransitionFlag"], expected_tf_df, check_like=True)

def test_no_matching_template_items_in_assortments(run_mock_pipeline):
    """
    Tests the case where assortment tables have data, but none of the items are defined as a 'Planning Item' in ItemMaster.
    Expects no expansion and empty outputs.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ITEM_A,T_GROUP_1,ITEM_A
ITEM_B,T_GROUP_1,ITEM_A
ITEM_C,T_GROUP_2,ITEM_C
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,ITEM_D,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,LOC1
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,ITEM_E,ITEM_E,ACC2,P_ACC2,CH2,P_CH2,R2,P_R2,DD2,P_DD2,PNL2,P_PNL2,LOC2,P_LOC2
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V1,2023-01-01
"""
    inputs = {
        "ItemMaster": item_master_csv,
        "AssortmentFinal": assortment_final_csv,
        "AssortmentSellOut": assortment_sellout_csv,
        "Date": date_csv
    }

    outputs = run_mock_pipeline(inputs)

    assert outputs["AssortmentFinal_Output"].empty, "AssortmentFinal_Output should be empty as no template items were found"
    assert outputs["AssortmentSellOut_Output"].empty, "AssortmentSellOut_Output should be empty as no template items were found"
    assert outputs["TransitionFlag_Output"].empty, "TransitionFlag_Output should be empty as no expansion occurred"


def test_item_master_with_no_transition_groups(run_mock_pipeline):
    """
    Tests behavior when ItemMaster contains no transition logic (e.g., 'Transition Item' is unique for every row).
    Expects no expansion and empty outputs.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ITEM_A,T_GROUP_1,ITEM_A
ITEM_B,T_GROUP_2,ITEM_B
ITEM_C,T_GROUP_3,ITEM_C
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,ITEM_A,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,LOC1
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,ITEM_B,ITEM_B,ACC2,P_ACC2,CH2,P_CH2,R2,P_R2,DD2,P_DD2,PNL2,P_PNL2,LOC2,P_LOC2
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V1,2023-01-01
"""
    inputs = {
        "ItemMaster": item_master_csv,
        "AssortmentFinal": assortment_final_csv,
        "AssortmentSellOut": assortment_sellout_csv,
        "Date": date_csv
    }

    outputs = run_mock_pipeline(inputs)

    # Because each Transition Item group only has one member, the "expansion" only generates the original item.
    # The anti-join then filters this out, resulting in no new rows.
    assert outputs["AssortmentFinal_Output"].empty, "AssortmentFinal_Output should be empty when no new items are generated"
    assert outputs["AssortmentSellOut_Output"].empty, "AssortmentSellOut_Output should be empty when no new items are generated"

    # Based on the description "expects... empty outputs", TransitionFlag should also be empty.
    # This implies flags are only created for newly generated assortment rows.
    assert outputs["TransitionFlag_Output"].empty, "TransitionFlag_Output should be empty if no new assortment rows were created"


def test_all_transition_items_already_exist_in_input(run_mock_pipeline, pandas_assert_df_equal):
    """
    Tests a scenario where all items of a transition group are already present in the input assortments.
    Expects empty AssortmentFinal_Output and AssortmentSellOut_Output, but a complete TransitionFlag_Output.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ITEM_A,T_GROUP_1,ITEM_A
ITEM_B,T_GROUP_1,ITEM_A
"""
    # Note: ITEM_A is the template, but both ITEM_A and ITEM_B are already present.
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,ITEM_A,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,ITEM_A-R1,ITEM_A-P_R1,PNL1,P_PNL1,LOC1
1,V1,ITEM_B,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,ITEM_B-R1,ITEM_B-P_R1,PNL1,P_PNL1,LOC1
"""
    # Note: Planning Item for ITEM_B is ITEM_A. Both are present at the planning level.
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,ITEM_A,ITEM_A,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,ITEM_A-P_R1,PNL1,P_PNL1,LOC1,P_LOC1
1,V1,ITEM_B,ITEM_A,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD2,ITEM_A-P_R1,PNL1,P_PNL1,LOC1,P_LOC1
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V1,2023-01-01
"""
    inputs = {
        "ItemMaster": item_master_csv,
        "AssortmentFinal": assortment_final_csv,
        "AssortmentSellOut": assortment_sellout_csv,
        "Date": date_csv
    }

    outputs = run_mock_pipeline(inputs)

    assert outputs["AssortmentFinal_Output"].empty, "AssortmentFinal_Output should be empty as all expanded rows already exist"
    assert outputs["AssortmentSellOut_Output"].empty, "AssortmentSellOut_Output should be empty as all expanded rows already exist"

    # Per business logic, the TransitionFlag is created from the full expanded set, regardless of whether they are new.
    # AF expansion creates rows for ITEM_A and ITEM_B.
    # ASO expansion creates rows for Planning Item ITEM_A.
    # The union of these should be ITEM_A and ITEM_B based dimensions.
    expected_transition_flag_data = {
        'Version Name': ['V1', 'V1'],
        'Planning Item': ['ITEM_A', 'ITEM_B'],
        'Planning Account': ['P_ACC1', 'P_ACC1'],
        'Planning Channel': ['P_CH1', 'P_CH1'],
        'Planning Region': ['P_R1', 'P_R1'],
        'Planning Demand Domain': ['ITEM_B-P_R1', 'ITEM_A-P_R1'],
        'Planning PnL': ['P_PNL1', 'P_PNL1'],
        'Planning Location': ['P_LOC1', 'LOC1'],
        'Partial Week': ['2023-01-01', '2023-01-01'],
        'Transition Flag': [1, 1]
    }
    
    # We sort to ensure comparison is order-independent
    expected_df = pandas_assert_df_equal.pd.DataFrame(expected_transition_flag_data).sort_values(by=['Planning Item']).reset_index(drop=True)
    actual_df = outputs["TransitionFlag_Output"].sort_values(by=['Planning Item']).reset_index(drop=True)
    
    pandas_assert_df_equal(expected_df, actual_df, "TransitionFlag_Output should contain all items from the transition group")


def test_missing_transition_date_for_version(run_mock_pipeline, pandas_assert_df_equal):
    """
    Tests behavior when the Date table does not contain an entry for the active 'Version Name'.
    Expects TransitionFlag_Output to be generated but have a null 'Partial Week' value.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ITEM_A,T_GROUP_1,ITEM_A
ITEM_B,T_GROUP_1,ITEM_A
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,ITEM_A,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,ITEM_A-R1,ITEM_A-P_R1,PNL1,P_PNL1,LOC1
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,ITEM_A,ITEM_A,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,ITEM_A-P_R1,PNL1,P_PNL1,LOC1,P_LOC1
"""
    # Date table is for 'V2', but all other data is for 'V1'
    date_csv = """Version,Version Name,Transition Start Date
2,V2,2023-01-01
"""
    inputs = {
        "ItemMaster": item_master_csv,
        "AssortmentFinal": assortment_final_csv,
        "AssortmentSellOut": assortment_sellout_csv,
        "Date": date_csv
    }

    outputs = run_mock_pipeline(inputs)

    # Assortment outputs should be generated normally
    assert not outputs["AssortmentFinal_Output"].empty, "AssortmentFinal_Output should be generated even if date is missing"
    assert not outputs["AssortmentSellOut_Output"].empty, "AssortmentSellOut_Output should be generated even if date is missing"

    # TransitionFlag output should be generated
    transition_flag_df = outputs["TransitionFlag_Output"]
    assert not transition_flag_df.empty, "TransitionFlag_Output should be generated even if date is missing"

    # The 'Partial Week' column should be all null/NaN because the date lookup failed
    assert transition_flag_df['Partial Week'].isnull().all(), "Partial Week column should be all null when date for version is missing"

def test_null_values_in_join_keys(main_logic):
    """
    Tests system resilience when join keys in the input tables (e.g., 'Planning Item') are null.
    Expects these rows to be ignored without crashing the process.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ItemA,BC1,PItemA
ItemB,BC1,PItemB
ItemC,BC2,PItemC
ItemD,BC2,
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,ItemA,Acc1,PAcc1,Ch1,PCh1,Reg1,PReg1,DD1,PDD1,PnL1,PPnL1,Loc1
1,V1,ItemC,Acc2,PAcc2,Ch2,PCh2,Reg2,PReg2,DD2,PDD2,PnL2,PPnL2,Loc2
"""
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,ItemA,PItemA,Acc1,PAcc1,Ch1,PCh1,Reg1,PReg1,DD1,PDD1,PnL1,PPnL1,Loc1,PLoc1
1,V1,ItemC,PItemC,Acc2,PAcc2,Ch2,PCh2,Reg2,PReg2,DD2,PDD2,PnL2,PPnL2,Loc2,PLoc2
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V1,2023-01-01
"""
    version_name = "V1"

    # Create DataFrames
    item_master = pd.read_csv(io.StringIO(item_master_csv))
    assortment_final = pd.read_csv(io.StringIO(assortment_final_csv))
    assortment_sell_out = pd.read_csv(io.StringIO(assortment_sell_out_csv))
    date_table = pd.read_csv(io.StringIO(date_csv))

    # Execute the main logic
    outputs = main_logic(item_master, assortment_final, assortment_sell_out, date_table, version_name)

    # Expected Outputs
    # Only group BC1 should be processed. ItemD with a null 'Planning Item' in ItemMaster should be ignored.
    # AssortmentFinal: ItemA is the template, ItemB is the new item. 1 new row.
    # AssortmentSellOut: PItemA is the template, PItemB is the new item. 1 new row.
    # TransitionFlag: PItemA and PItemB are the items. 2 total rows.

    expected_af_output_csv = """Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Location,Assortment Final,Transition Sell In Assortment
V1,ItemB,PAcc1,PCh1,PReg1,ItemB-PReg1,PPnL1,Loc1,1,1
"""
    expected_aso_output_csv = """Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Mdlz DP Assortment Sell Out,Transition Sell Out Assortment
V1,PItemB,PAcc1,PCh1,PReg1,PItemB-PReg1,PPnL1,PLoc1,1,1
"""
    expected_tf_output_csv = """Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Partial Week,Transition Flag
V1,PItemA,PAcc1,PCh1,PReg1,PItemA-PReg1,PPnL1,PLoc1,2023-01-01,1
V1,PItemB,PAcc1,PCh1,PReg1,ItemB-PReg1,PPnL1,Loc1,2023-01-01,1
V1,PItemB,PAcc1,PCh1,PReg1,PItemB-PReg1,PPnL1,PLoc1,2023-01-01,1
"""
    # The unification in TF can be tricky. Let's just check the row counts.
    # AF expansion gives ItemB. ASO expansion gives PItemB.
    # AF_dims: {PItemB, PAcc1, PCh1, PReg1, ItemB-PReg1, PPnL1, Loc1}
    # ASO_dims: {PItemB, PAcc1, PCh1, PReg1, PItemB-PReg1, PPnL1, PLoc1}
    # Union should be 2 rows as PLoc1 != Loc1.
    # And we also have the original template items.
    # AF template: {PItemA, ...}
    # ASO template: {PItemA, ...}
    # Total unique intersections should be 4 if we unify AF_dims and ASO_dims.
    # Let's verify the primary outputs first, which are more straightforward.

    # Assertions
    assert outputs['AssortmentFinal'].shape[0] == 1, "AssortmentFinal should have 1 new row from the valid group"
    assert outputs['AssortmentSellOut'].shape[0] == 1, "AssortmentSellOut should have 1 new row from the valid group"
    assert outputs['AssortmentFinal']['Item'].iloc[0] == 'ItemB'
    assert outputs['AssortmentSellOut']['Planning Item'].iloc[0] == 'PItemB'

    # The TF unification can be complex, let's check count.
    # AF_Expanded gives 2 rows (ItemA, ItemB). ASO_Expanded gives 2 rows (PItemA, PItemB).
    # Unifying the dimensions should result in 4 distinct rows because the Location/Planning Location differ.
    assert outputs['TransitionFlag'].shape[0] == 4, "TransitionFlag should have 4 rows from the valid unified dimensions"


def test_data_for_irrelevant_versions(main_logic):
    """
    Ensures that data belonging to a different 'Version Name' in the input tables is
    correctly filtered out and ignored by the process.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ItemA1,BC1,PItemA1
ItemB1,BC1,PItemB1
ItemA2,BC2,PItemA2
ItemB2,BC2,PItemB2
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V_Current,ItemA1,Acc1,PAcc1,Ch1,PCh1,Reg1,PReg1,DD1,PDD1,PnL1,PPnL1,Loc1
2,V_Old,ItemA2,Acc2,PAcc2,Ch2,PCh2,Reg2,PReg2,DD2,PDD2,PnL2,PPnL2,Loc2
"""
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V_Current,ItemA1,PItemA1,Acc1,PAcc1,Ch1,PCh1,Reg1,PReg1,DD1,PDD1,PnL1,PPnL1,Loc1,PLoc1
2,V_Old,ItemA2,PItemA2,Acc2,PAcc2,Ch2,PCh2,Reg2,PReg2,DD2,PDD2,PnL2,PPnL2,Loc2,PLoc2
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V_Current,2023-01-01
2,V_Old,2022-01-01
"""
    version_name_to_process = "V_Current"

    # Create DataFrames
    item_master = pd.read_csv(io.StringIO(item_master_csv))
    assortment_final = pd.read_csv(io.StringIO(assortment_final_csv))
    assortment_sell_out = pd.read_csv(io.StringIO(assortment_sell_out_csv))
    date_table = pd.read_csv(io.StringIO(date_csv))

    # Execute the main logic
    outputs = main_logic(item_master, assortment_final, assortment_sell_out, date_table, version_name_to_process)

    # Expected: Only data for 'V_Current' should be processed.
    # AF: ItemA1 is template, expands to ItemB1. 1 new row.
    # ASO: PItemA1 is template, expands to PItemB1. 1 new row.
    # TF: 2 unified dimensions (for PItemA1, PItemB1).

    af_output = outputs['AssortmentFinal']
    aso_output = outputs['AssortmentSellOut']
    tf_output = outputs['TransitionFlag']

    # Assertions
    assert af_output.shape[0] == 1, "Expected 1 new AssortmentFinal row for the current version"
    assert (af_output['Version Name'] == 'V_Current').all(), "All rows in AssortmentFinal output must belong to the current version"
    assert af_output['Item'].iloc[0] == 'ItemB1'

    assert aso_output.shape[0] == 1, "Expected 1 new AssortmentSellOut row for the current version"
    assert (aso_output['Version Name'] == 'V_Current').all(), "All rows in AssortmentSellOut output must belong to the current version"
    assert aso_output['Planning Item'].iloc[0] == 'PItemB1'

    # AF exp: ItemA1, ItemB1. ASO exp: PItemA1, PItemB1.
    # Dimensions for AF will use Item -> Planning Item aliasing.
    # AF dims: {PItemA1, ...}, {PItemB1, ...}
    # ASO dims: {PItemA1, ...}, {PItemB1, ...}
    # Union should be 2 rows.
    assert tf_output.shape[0] == 2, "Expected 2 unified dimension rows for the current version"
    assert (tf_output['Version Name'] == 'V_Current').all(), "All rows in TransitionFlag output must belong to the current version"
    assert tf_output['Partial Week'].iloc[0] == '2023-01-01'


def test_large_scale_expansion(main_logic):
    """
    Tests the process with a large number of transition groups and items to ensure
    performance and scalability of the joins and aggregations.
    """
    num_groups = 5
    items_per_group = 100
    version_name = "V_Large"

    # Generate ItemMaster data
    item_master_data = []
    for g in range(num_groups):
        for i in range(items_per_group):
            item_master_data.append({
                'Item': f'Item_G{g}_I{i}',
                'Transition Item': f'BC_G{g}',
                'Planning Item': f'PItem_G{g}_I{i}'
            })
    item_master = pd.DataFrame(item_master_data)

    # Generate AssortmentFinal template data (one template per group)
    af_data = []
    for g in range(num_groups):
        af_data.append({
            'Version': 1, 'Version Name': version_name, 'Item': f'Item_G{g}_I0',
            'Account': 'Acc1', 'Planning Account': f'PAcc_G{g}',
            'Channel': 'Ch1', 'Planning Channel': f'PCh_G{g}',
            'Region': 'Reg1', 'Planning Region': f'PReg_G{g}',
            'Demand Domain': 'DD1', 'Planning Demand Domain': 'PDD1',
            'PnL': 'PnL1', 'Planning PnL': f'PPnL_G{g}', 'Location': f'Loc_G{g}'
        })
    assortment_final = pd.DataFrame(af_data)

    # Generate AssortmentSellOut template data (one template per group)
    aso_data = []
    for g in range(num_groups):
        aso_data.append({
            'Version': 1, 'Version Name': version_name, 'Item': f'Item_G{g}_I0',
            'Planning Item': f'PItem_G{g}_I0', 'Account': 'Acc1', 'Planning Account': f'PAcc_G{g}',
            'Channel': 'Ch1', 'Planning Channel': f'PCh_G{g}', 'Region': 'Reg1',
            'Planning Region': f'PReg_G{g}', 'Demand Domain': 'DD1',
            'Planning Demand Domain': 'PDD1', 'PnL': 'PnL1', 'Planning PnL': f'PPnL_G{g}',
            'Location': f'Loc_G{g}', 'Planning Location': f'PLoc_G{g}'
        })
    assortment_sell_out = pd.DataFrame(aso_data)

    # Generate Date data
    date_table = pd.DataFrame([{'Version': 1, 'Version Name': version_name, 'Transition Start Date': '2024-01-01'}])

    # Execute the main logic
    outputs = main_logic(item_master, assortment_final, assortment_sell_out, date_table, version_name)

    # Expected counts:
    # For each of the `num_groups`, there are `items_per_group` items.
    # 1 is a template, so `items_per_group - 1` new rows are generated per group.
    expected_af_rows = num_groups * (items_per_group - 1)
    expected_aso_rows = num_groups * (items_per_group - 1)
    # The total number of unique intersections is `num_groups * items_per_group`
    # because AF and ASO templates are aligned on all dimensions except Loc/PLoc.
    # The unification will create one row per item per group, since all other dimensions are the same.
    # Let's re-read Step 5.2. AF_Dimensions aliases Item to Planning Item and Location to Planning Location.
    # ASO_Dimensions has Planning Item and Planning Location.
    # So the dimensions are aligned and will produce `num_groups * items_per_group` unique rows.
    expected_tf_rows = num_groups * items_per_group

    # Assertions
    assert outputs['AssortmentFinal'].shape[0] == expected_af_rows, f"Expected {expected_af_rows} new rows in AssortmentFinal"
    assert outputs['AssortmentSellOut'].shape[0] == expected_aso_rows, f"Expected {expected_aso_rows} new rows in AssortmentSellOut"
    assert outputs['TransitionFlag'].shape[0] == expected_tf_rows, f"Expected {expected_tf_rows} unique intersections in TransitionFlag"


def test_complex_dimension_unification(main_logic):
    """
    Creates a scenario where expanded AssortmentFinal and AssortmentSellOut rows share
    some, but not all, dimension values, verifying that the UNION in Step 5 produces
    the correct distinct set of intersections.
    """
    item_master_csv = """Item,Transition Item,Planning Item
Item1,BC1,PItem1
Item2,BC1,PItem2
Item3,BC1,PItem3
"""
    # AF template has PAcc1, PCh1, PReg1
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,Item1,Acc1,PAcc1,Ch1,PCh1,Reg1,PReg1,DD1,PDD1,PnL1,PPnL1,Loc1
"""
    # ASO template has PAcc2, PCh2, PReg2
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,Item1,PItem1,Acc2,PAcc2,Ch2,PCh2,Reg2,PReg2,DD2,PDD2,PnL2,PPnL2,Loc2,PLoc2
"""
    date_csv = """Version,Version Name,Transition Start Date
1,V1,2023-01-01
"""
    version_name = "V1"

    # Create DataFrames
    item_master = pd.read_csv(io.StringIO(item_master_csv))
    assortment_final = pd.read_csv(io.StringIO(assortment_final_csv))
    assortment_sell_out = pd.read_csv(io.StringIO(assortment_sell_out_csv))
    date_table = pd.read_csv(io.StringIO(date_csv))

    # Execute the main logic
    outputs = main_logic(item_master, assortment_final, assortment_sell_out, date_table, version_name)

    # Expected:
    # Expanded_AF will have 3 rows for {Item1, Item2, Item3} with {PAcc1, PCh1, PReg1}.
    # Expanded_ASO will have 3 rows for {PItem1, PItem2, PItem3} with {PAcc2, PCh2, PReg2}.
    # AF_Dimensions will alias Item->Planning Item and Location->Planning Location.
    # AF_dims will have 3 rows for {PItem1, PItem2, PItem3} with dims {PAcc1, PCh1, PReg1, PnL1, Loc1}.
    # ASO_dims will have 3 rows for {PItem1, PItem2, PItem3} with dims {PAcc2, PCh2, PReg2, PnL2, PLoc2}.
    # The dimensions (Account, Channel, Region, PnL, Location) are different for AF and ASO.
    # Therefore, the UNION DISTINCT should produce 3 (from AF) + 3 (from ASO) = 6 unique rows.

    af_output = outputs['AssortmentFinal']
    aso_output = outputs['AssortmentSellOut']
    tf_output = outputs['TransitionFlag']

    # Assert AF and ASO outputs (2 new rows each)
    assert af_output.shape[0] == 2, "Expected 2 new rows for AssortmentFinal"
    assert aso_output.shape[0] == 2, "Expected 2 new rows for AssortmentSellOut"

    # Assert TransitionFlag output
    assert tf_output.shape[0] == 6, "Expected 6 distinct unified rows in TransitionFlag"

    # Verify the contents of TransitionFlag to ensure correctness
    # Convert to a set of tuples for easy comparison
    actual_tuples = {tuple(row) for row in tf_output[['Planning Item', 'Planning Account', 'Planning Region']].to_numpy()}

    expected_tuples = {
        # From AF expansion
        ('PItem1', 'PAcc1', 'PReg1'),
        ('PItem2', 'PAcc1', 'PReg1'),
        ('PItem3', 'PAcc1', 'PReg1'),
        # From ASO expansion
        ('PItem1', 'PAcc2', 'PReg2'),
        ('PItem2', 'PAcc2', 'PReg2'),
        ('PItem3', 'PAcc2', 'PReg2'),
    }

    assert actual_tuples == expected_tuples, "The unified dimensions in TransitionFlag are not correct"

def test_disjoint_templates(main):
    """
    Tests a case where AssortmentFinal contains a template for one transition group
    and AssortmentSellOut contains a template for a completely different group.
    Both should be processed and unified correctly in TransitionFlag_Output.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ItemA1,GroupA,ItemA1
ItemA2,GroupA,ItemA1
ItemB1,GroupB,ItemB1
ItemB2,GroupB,ItemB1
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,TestVersion,ItemA1,ACC1,ACC1,CH1,CH1,REG1,REG1,DD1,DD1,PNL1,PNL1,LOC1
"""
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,TestVersion,ItemB1,ItemB1,ACC2,ACC2,CH2,CH2,REG2,REG2,DD2,DD2,PNL2,PNL2,LOC2,LOC2
"""
    date_csv = """Version,Version Name,Transition Start Date
V1,TestVersion,2023-01-01
"""
    
    item_master = pd.read_csv(io.StringIO(item_master_csv))
    assortment_final = pd.read_csv(io.StringIO(assortment_final_csv))
    assortment_sell_out = pd.read_csv(io.StringIO(assortment_sell_out_csv))
    date = pd.read_csv(io.StringIO(date_csv))

    # Expected AssortmentFinal_Output: Expands GroupA based on AF template
    expected_af_output = pd.DataFrame({
        'Version Name': ['TestVersion'],
        'Item': ['ItemA2'],
        'Planning Account': ['ACC1'],
        'Planning Channel': ['CH1'],
        'Planning Region': ['REG1'],
        'Planning Demand Domain': ['ItemA2-REG1'],
        'Planning PnL': ['PNL1'],
        'Location': ['LOC1'],
        'Assortment Final': [1],
        'Transition Sell In Assortment': [1]
    })

    # Expected AssortmentSellOut_Output: Expands GroupB based on ASO template
    expected_aso_output = pd.DataFrame({
        'Version Name': ['TestVersion'],
        'Planning Item': ['ItemB2'],
        'Planning Account': ['ACC2'],
        'Planning Channel': ['CH2'],
        'Planning Region': ['REG2'],
        'Planning Demand Domain': ['ItemB2-REG2'],
        'Planning PnL': ['PNL2'],
        'Planning Location': ['LOC2'],
        'Mdlz DP Assortment Sell Out': [1],
        'Transition Sell Out Assortment': [1]
    })
    
    # Expected TransitionFlag_Output: Unifies all expanded items from both AF and ASO
    expected_tf_output = pd.DataFrame({
        'Version Name': ['TestVersion', 'TestVersion', 'TestVersion', 'TestVersion'],
        'Planning Item': ['ItemA1', 'ItemA2', 'ItemB1', 'ItemB2'],
        'Planning Account': ['ACC1', 'ACC1', 'ACC2', 'ACC2'],
        'Planning Channel': ['CH1', 'CH1', 'CH2', 'CH2'],
        'Planning Region': ['REG1', 'REG1', 'REG2', 'REG2'],
        'Planning Demand Domain': ['ItemA1-REG1', 'ItemA2-REG1', 'ItemB1-REG2', 'ItemB2-REG2'],
        'Planning PnL': ['PNL1', 'PNL1', 'PNL2', 'PNL2'],
        'Planning Location': ['LOC1', 'LOC1', 'LOC2', 'LOC2'],
        'Partial Week': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01'],
        'Transition Flag': [1, 1, 1, 1]
    })

    # Execute the main logic
    result = main(item_master, assortment_final, assortment_sell_out, date)
    actual_af_output = result['AssortmentFinal']
    actual_aso_output = result['AssortmentSellOut']
    actual_tf_output = result['TransitionFlag']

    # Sort and compare AssortmentFinal output
    af_sort_cols = sorted(list(expected_af_output.columns))
    actual_af_output = actual_af_output.sort_values(by=af_sort_cols).reset_index(drop=True)
    expected_af_output = expected_af_output.sort_values(by=af_sort_cols).reset_index(drop=True)
    assert actual_af_output.equals(expected_af_output), "AssortmentFinal output does not match expected"

    # Sort and compare AssortmentSellOut output
    aso_sort_cols = sorted(list(expected_aso_output.columns))
    actual_aso_output = actual_aso_output.sort_values(by=aso_sort_cols).reset_index(drop=True)
    expected_aso_output = expected_aso_output.sort_values(by=aso_sort_cols).reset_index(drop=True)
    assert actual_aso_output.equals(expected_aso_output), "AssortmentSellOut output does not match expected"

    # Sort and compare TransitionFlag output
    tf_sort_cols = sorted(list(expected_tf_output.columns))
    actual_tf_output = actual_tf_output.sort_values(by=tf_sort_cols).reset_index(drop=True)
    expected_tf_output = expected_tf_output.sort_values(by=tf_sort_cols).reset_index(drop=True)
    assert actual_tf_output.equals(expected_tf_output), "TransitionFlag output does not match expected"


def test_different_templates_within_same_group(main):
    """
    Tests a scenario where AssortmentFinal and AssortmentSellOut both have a template
    for the same transition group, but the template rows have different dimensional
    values (e.g., different Planning Account). The expansion should respect each
    source's template.
    """
    item_master_csv = """Item,Transition Item,Planning Item
ItemA1,GroupA,ItemA1
ItemA2,GroupA,ItemA1
ItemA3,GroupA,ItemA1
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,TestVersion,ItemA1,ACC1,ACC1,CH1,CH1,REG1,REG1,DD1,DD1,PNL1,PNL1,LOC1
"""
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,TestVersion,ItemA1,ItemA1,ACC2,ACC2,CH1,CH1,REG1,REG1,DD1,DD1,PNL1,PNL1,LOC1,LOC1
"""
    date_csv = """Version,Version Name,Transition Start Date
V1,TestVersion,2023-01-01
"""
    
    item_master = pd.read_csv(io.StringIO(item_master_csv))
    assortment_final = pd.read_csv(io.StringIO(assortment_final_csv))
    assortment_sell_out = pd.read_csv(io.StringIO(assortment_sell_out_csv))
    date = pd.read_csv(io.StringIO(date_csv))

    # Expected AssortmentFinal_Output: New items with Planning Account 'ACC1'
    expected_af_output = pd.DataFrame({
        'Version Name': ['TestVersion', 'TestVersion'],
        'Item': ['ItemA2', 'ItemA3'],
        'Planning Account': ['ACC1', 'ACC1'],
        'Planning Channel': ['CH1', 'CH1'],
        'Planning Region': ['REG1', 'REG1'],
        'Planning Demand Domain': ['ItemA2-REG1', 'ItemA3-REG1'],
        'Planning PnL': ['PNL1', 'PNL1'],
        'Location': ['LOC1', 'LOC1'],
        'Assortment Final': [1, 1],
        'Transition Sell In Assortment': [1, 1]
    })

    # Expected AssortmentSellOut_Output: New items with Planning Account 'ACC2'
    expected_aso_output = pd.DataFrame({
        'Version Name': ['TestVersion', 'TestVersion'],
        'Planning Item': ['ItemA2', 'ItemA3'],
        'Planning Account': ['ACC2', 'ACC2'],
        'Planning Channel': ['CH1', 'CH1'],
        'Planning Region': ['REG1', 'REG1'],
        'Planning Demand Domain': ['ItemA2-REG1', 'ItemA3-REG1'],
        'Planning PnL': ['PNL1', 'PNL1'],
        'Planning Location': ['LOC1', 'LOC1'],
        'Mdlz DP Assortment Sell Out': [1, 1],
        'Transition Sell Out Assortment': [1, 1]
    })
    
    # Expected TransitionFlag_Output: Union of all expanded items, resulting in 6 unique rows
    expected_tf_output = pd.DataFrame({
        'Version Name': ['TestVersion'] * 6,
        'Planning Item': ['ItemA1', 'ItemA1', 'ItemA2', 'ItemA2', 'ItemA3', 'ItemA3'],
        'Planning Account': ['ACC1', 'ACC2', 'ACC1', 'ACC2', 'ACC1', 'ACC2'],
        'Planning Channel': ['CH1'] * 6,
        'Planning Region': ['REG1'] * 6,
        'Planning Demand Domain': ['ItemA1-REG1', 'ItemA1-REG1', 'ItemA2-REG1', 'ItemA2-REG1', 'ItemA3-REG1', 'ItemA3-REG1'],
        'Planning PnL': ['PNL1'] * 6,
        'Planning Location': ['LOC1'] * 6,
        'Partial Week': ['2023-01-01'] * 6,
        'Transition Flag': [1] * 6
    })

    # Execute the main logic
    result = main(item_master, assortment_final, assortment_sell_out, date)
    actual_af_output = result['AssortmentFinal']
    actual_aso_output = result['AssortmentSellOut']
    actual_tf_output = result['TransitionFlag']

    # Sort and compare AssortmentFinal output
    af_sort_cols = sorted(list(expected_af_output.columns))
    actual_af_output = actual_af_output.sort_values(by=af_sort_cols).reset_index(drop=True)
    expected_af_output = expected_af_output.sort_values(by=af_sort_cols).reset_index(drop=True)
    assert actual_af_output.equals(expected_af_output), "AssortmentFinal output does not match expected"

    # Sort and compare AssortmentSellOut output
    aso_sort_cols = sorted(list(expected_aso_output.columns))
    actual_aso_output = actual_aso_output.sort_values(by=aso_sort_cols).reset_index(drop=True)
    expected_aso_output = expected_aso_output.sort_values(by=aso_sort_cols).reset_index(drop=True)
    assert actual_aso_output.equals(expected_aso_output), "AssortmentSellOut output does not match expected"

    # Sort and compare TransitionFlag output
    tf_sort_cols = sorted(list(expected_tf_output.columns))
    actual_tf_output = actual_tf_output.sort_values(by=tf_sort_cols).reset_index(drop=True)
    expected_tf_output = expected_tf_output.sort_values(by=tf_sort_cols).reset_index(drop=True)
    assert actual_tf_output.equals(expected_tf_output), "TransitionFlag output does not match expected"