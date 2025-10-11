from helpers.plugin_module import main
import io
from io import StringIO
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


def test_golden_record_basic_expansion():
    """
    Validates the core success path where a single row in AssortmentFinal and AssortmentSellOut,
    linked to a Transition Item with multiple SKUs, correctly generates new rows for the other
    SKUs in the same transition group.
    """
    # Input DataFrames
    item_master_df = pd.DataFrame([
        {"Item": "SKU101", "Transition Item": "BASE1", "Planning Item": "P1"},
        {"Item": "SKU102", "Transition Item": "BASE1", "Planning Item": "P2"},
        {"Item": "SKU103", "Transition Item": "BASE1", "Planning Item": "P3"},
        {"Item": "SKU201", "Transition Item": "BASE2", "Planning Item": "P4"},
    ])
    assortment_final_df = pd.DataFrame([
        {
            "Version": "V1", "Version Name": "Actuals", "Item": "SKU101",
            "Account": "ACC1", "Planning Account": "PACC1", "Channel": "CH1", "Planning Channel": "PCH1",
            "Region": "R1", "Planning Region": "PR1", "Demand Domain": "DD1", "Planning Demand Domain": "PDD_OLD",
            "PnL": "PnL1", "Planning PnL": "PPNL1", "Location": "LOC1"
        }
    ])
    assortment_sellout_df = pd.DataFrame([
        {
            "Version": "V1", "Version Name": "Actuals", "Item": "I1", "Planning Item": "SKU101",
            "Account": "ACC1", "Planning Account": "PACC1", "Channel": "CH1", "Planning Channel": "PCH1",
            "Region": "R1", "Planning Region": "PR1", "Demand Domain": "DD1", "Planning Demand Domain": "PDD_OLD",
            "PnL": "PnL1", "Planning PnL": "PPNL1", "Location": "LOC1", "Planning Location": "PLOC1"
        }
    ])
    date_df = pd.DataFrame([{"Version": "V1", "Version Name": "Actuals", "Transition Start Date": "2023-10-26"}])

    # Execute the transformation
    actual_outputs = main({
        "ItemMaster": item_master_df,
        "AssortmentFinal": assortment_final_df,
        "AssortmentSellOut": assortment_sellout_df,
        "Date": date_df
    })

    # Expected Outputs
    expected_assortment_final = pd.DataFrame([
        {
            "Version Name": "Actuals", "Item": "SKU102", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU102-PR1", "Planning PnL": "PPNL1", "Location": "LOC1",
            "Assortment Final": 1, "Transition Sell In Assortment": 1
        },
        {
            "Version Name": "Actuals", "Item": "SKU103", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU103-PR1", "Planning PnL": "PPNL1", "Location": "LOC1",
            "Assortment Final": 1, "Transition Sell In Assortment": 1
        }
    ])

    expected_assortment_sellout = pd.DataFrame([
        {
            "Version Name": "Actuals", "Planning Item": "SKU102", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU102-PR1", "Planning PnL": "PPNL1", "Planning Location": "PLOC1",
            "Mdlz DP Assortment Sell Out": 1, "Transition Sell Out Assortment": 1
        },
        {
            "Version Name": "Actuals", "Planning Item": "SKU103", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU103-PR1", "Planning PnL": "PPNL1", "Planning Location": "PLOC1",
            "Mdlz DP Assortment Sell Out": 1, "Transition Sell Out Assortment": 1
        }
    ])

    expected_transition_flag = pd.DataFrame([
        {
            "Version Name": "Actuals", "Planning Item": "SKU101", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU101-PR1", "Planning PnL": "PPNL1", "Planning Location": "LOC1",
            "Partial Week": "2023-10-26", "Transition Flag": 1
        },
        {
            "Version Name": "Actuals", "Planning Item": "SKU102", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU102-PR1", "Planning PnL": "PPNL1", "Planning Location": "LOC1",
            "Partial Week": "2023-10-26", "Transition Flag": 1
        },
        {
            "Version Name": "Actuals", "Planning Item": "SKU103", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU103-PR1", "Planning PnL": "PPNL1", "Planning Location": "LOC1",
            "Partial Week": "2023-10-26", "Transition Flag": 1
        },
        {
            "Version Name": "Actuals", "Planning Item": "SKU101", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU101-PR1", "Planning PnL": "PPNL1", "Planning Location": "PLOC1",
            "Partial Week": "2023-10-26", "Transition Flag": 1
        },
        {
            "Version Name": "Actuals", "Planning Item": "SKU102", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU102-PR1", "Planning PnL": "PPNL1", "Planning Location": "PLOC1",
            "Partial Week": "2023-10-26", "Transition Flag": 1
        },
        {
            "Version Name": "Actuals", "Planning Item": "SKU103", "Planning Account": "PACC1", "Planning Channel": "PCH1",
            "Planning Region": "PR1", "Planning Demand Domain": "SKU103-PR1", "Planning PnL": "PPNL1", "Planning Location": "PLOC1",
            "Partial Week": "2023-10-26", "Transition Flag": 1
        }
    ])

    # Assertions
    sort_keys_final = ["Item", "Planning Account", "Planning Channel"]
    actual_final_list = actual_outputs['AssortmentFinal'].sort_values(by=sort_keys_final).to_dict('records')
    expected_final_list = expected_assortment_final.sort_values(by=sort_keys_final).to_dict('records')
    assert actual_final_list == expected_final_list

    sort_keys_sellout = ["Planning Item", "Planning Account", "Planning Channel"]
    actual_sellout_list = actual_outputs['AssortmentSellOut'].sort_values(by=sort_keys_sellout).to_dict('records')
    expected_sellout_list = expected_assortment_sellout.sort_values(by=sort_keys_sellout).to_dict('records')
    assert actual_sellout_list == expected_sellout_list

    sort_keys_flag = ["Planning Item", "Planning Location"]
    actual_flag_list = actual_outputs['TransitionFlag'].sort_values(by=sort_keys_flag).to_dict('records')
    expected_flag_list = expected_transition_flag.sort_values(by=sort_keys_flag).to_dict('records')
    assert actual_flag_list == expected_flag_list


def test_output_schema_and_static_values():
    """
    Verifies that all three output tables conform to the specified schema and that the static
    flag columns (e.g., 'Assortment Final', 'Transition Flag') are correctly populated with the value '1'.
    """
    # Input DataFrames
    item_master_df = pd.DataFrame([
        {"Item": "SKU1", "Transition Item": "BASE1", "Planning Item": "P1"},
        {"Item": "SKU2", "Transition Item": "BASE1", "Planning Item": "P2"},
    ])
    assortment_final_df = pd.DataFrame([{"Item": "SKU1", "Planning Account": "PA1", "Planning Channel": "PC1", "Planning Region": "PR1", "Planning PnL": "PPNL1", "Location": "L1", "Version Name": "VN1"}])
    assortment_sellout_df = pd.DataFrame([{"Planning Item": "SKU1", "Planning Account": "PA1", "Planning Channel": "PC1", "Planning Region": "PR1", "Planning PnL": "PPNL1", "Planning Location": "PL1", "Version Name": "VN1"}])
    date_df = pd.DataFrame([{"Transition Start Date": "2023-01-01"}])

    # Execute the transformation
    actual_outputs = main({
        "ItemMaster": item_master_df,
        "AssortmentFinal": assortment_final_df,
        "AssortmentSellOut": assortment_sellout_df,
        "Date": date_df
    })
    
    # Expected Schemas
    expected_schema_final = sorted([
        "Version Name", "Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Location", "Assortment Final",
        "Transition Sell In Assortment"
    ])
    expected_schema_sellout = sorted([
        "Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Planning Location", "Mdlz DP Assortment Sell Out",
        "Transition Sell Out Assortment"
    ])
    expected_schema_flag = sorted([
        "Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Planning Location", "Partial Week", "Transition Flag"
    ])

    # Assertions for schema
    assert sorted(actual_outputs['AssortmentFinal'].columns.tolist()) == expected_schema_final
    assert sorted(actual_outputs['AssortmentSellOut'].columns.tolist()) == expected_schema_sellout
    assert sorted(actual_outputs['TransitionFlag'].columns.tolist()) == expected_schema_flag

    # Assertions for static values (ensure tables are not empty before checking all())
    assert not actual_outputs['AssortmentFinal'].empty
    assert (actual_outputs['AssortmentFinal']['Assortment Final'] == 1).all()
    assert (actual_outputs['AssortmentFinal']['Transition Sell In Assortment'] == 1).all()

    assert not actual_outputs['AssortmentSellOut'].empty
    assert (actual_outputs['AssortmentSellOut']['Mdlz DP Assortment Sell Out'] == 1).all()
    assert (actual_outputs['AssortmentSellOut']['Transition Sell Out Assortment'] == 1).all()

    assert not actual_outputs['TransitionFlag'].empty
    assert (actual_outputs['TransitionFlag']['Transition Flag'] == 1).all()


def test_planning_demand_domain_concatenation():
    """
    Confirms that the 'Planning Demand Domain' in all expanded rows is correctly generated by
    concatenating the new item/SKU and the original 'Planning Region' from the source row.
    """
    # Input DataFrames
    item_master_df = pd.DataFrame([
        {"Item": "SKU_A", "Transition Item": "BASE_C", "Planning Item": "PA"},
        {"Item": "SKU_B", "Transition Item": "BASE_C", "Planning Item": "PB"},
    ])
    assortment_final_df = pd.DataFrame([{
        "Version Name": "V_CONCAT", "Item": "SKU_A", "Planning Account": "PACC",
        "Planning Channel": "PCHAN", "Planning Region": "REGION_FINAL",
        "Planning PnL": "PPNL", "Location": "LOC_F"
    }])
    assortment_sellout_df = pd.DataFrame([{
        "Version Name": "V_CONCAT", "Planning Item": "SKU_A", "Planning Account": "PACC",
        "Planning Channel": "PCHAN", "Planning Region": "REGION_SELLOUT",
        "Planning PnL": "PPNL", "Planning Location": "PLOC_S"
    }])
    date_df = pd.DataFrame([{"Transition Start Date": "2023-01-01"}])
    
    # Execute the transformation
    actual_outputs = main({
        "ItemMaster": item_master_df,
        "AssortmentFinal": assortment_final_df,
        "AssortmentSellOut": assortment_sellout_df,
        "Date": date_df
    })

    # Expected Outputs with correct concatenation
    expected_assortment_final = pd.DataFrame([{
        "Version Name": "V_CONCAT", "Item": "SKU_B", "Planning Account": "PACC", "Planning Channel": "PCHAN",
        "Planning Region": "REGION_FINAL", "Planning Demand Domain": "SKU_B-REGION_FINAL", "Planning PnL": "PPNL",
        "Location": "LOC_F", "Assortment Final": 1, "Transition Sell In Assortment": 1
    }])
    expected_assortment_sellout = pd.DataFrame([{
        "Version Name": "V_CONCAT", "Planning Item": "SKU_B", "Planning Account": "PACC", "Planning Channel": "PCHAN",
        "Planning Region": "REGION_SELLOUT", "Planning Demand Domain": "SKU_B-REGION_SELLOUT", "Planning PnL": "PPNL",
        "Planning Location": "PLOC_S", "Mdlz DP Assortment Sell Out": 1, "Transition Sell Out Assortment": 1
    }])
    
    # Assertions
    actual_final_list = actual_outputs['AssortmentFinal'].to_dict('records')
    expected_final_list = expected_assortment_final.to_dict('records')
    assert actual_final_list == expected_final_list
    assert actual_final_list[0]['Planning Demand Domain'] == 'SKU_B-REGION_FINAL'

    actual_sellout_list = actual_outputs['AssortmentSellOut'].to_dict('records')
    expected_sellout_list = expected_assortment_sellout.to_dict('records')
    assert actual_sellout_list == expected_sellout_list
    assert actual_sellout_list[0]['Planning Demand Domain'] == 'SKU_B-REGION_SELLOUT'
    
    # Check concatenation in TransitionFlag output as well
    flag_df = actual_outputs['TransitionFlag']
    domain_for_sku_b_final = flag_df[
        (flag_df['Planning Item'] == 'SKU_B') & (flag_df['Planning Location'] == 'LOC_F')
    ]['Planning Demand Domain'].iloc[0]
    domain_for_sku_b_sellout = flag_df[
        (flag_df['Planning Item'] == 'SKU_B') & (flag_df['Planning Location'] == 'PLOC_S')
    ]['Planning Demand Domain'].iloc[0]

    assert domain_for_sku_b_final == "SKU_B-REGION_FINAL"
    assert domain_for_sku_b_sellout == "SKU_B-REGION_SELLOUT"


def test_transition_date_propagation():
    """
    Ensures the 'Partial Week' column in the 'TransitionFlag' output is correctly populated
    with the single 'Transition Start Date' value from the 'Date' input table for all rows.
    """
    # Input DataFrames with a distinct date
    test_date = '2099-12-31'
    item_master_df = pd.DataFrame([
        {"Item": "SKU1", "Transition Item": "BASE1", "Planning Item": "P1"},
        {"Item": "SKU2", "Transition Item": "BASE1", "Planning Item": "P2"},
    ])
    assortment_final_df = pd.DataFrame([{"Item": "SKU1", "Planning Account": "PA1", "Planning Channel": "PC1", "Planning Region": "PR1", "Planning PnL": "PPNL1", "Location": "L1", "Version Name": "VN1"}])
    assortment_sellout_df = pd.DataFrame([]) # Test with an empty table to ensure union still works
    date_df = pd.DataFrame([{"Transition Start Date": test_date}])

    # Execute the transformation
    actual_outputs = main({
        "ItemMaster": item_master_df,
        "AssortmentFinal": assortment_final_df,
        "AssortmentSellOut": assortment_sellout_df,
        "Date": date_df
    })

    output_flag = actual_outputs['TransitionFlag']

    # Assertions
    # 1. Ensure the output is not empty
    assert not output_flag.empty, "TransitionFlag output should not be empty."
    # 2. Check that the 'Partial Week' column exists
    assert 'Partial Week' in output_flag.columns
    # 3. Verify all values in the 'Partial Week' column match the input date
    assert (output_flag['Partial Week'] == test_date).all(), f"All 'Partial Week' values should be '{test_date}'"

def test_multiple_distinct_transition_groups():
    """
    Tests a scenario with multiple 'Transition Item' groups in ItemMaster and verifies that
    assortments for items in one group are only expanded to other items within that same group,
    without cross-contamination.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU1,T_ITEM1,P_SKU1
SKU2,T_ITEM1,P_SKU2
SKU3,T_ITEM2,P_SKU3
SKU4,T_ITEM2,P_SKU4
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,VN1,SKU1,ACC1,P_ACC1,CH1,P_CH1,REG1,P_REG1,DD1,P_DD1,PNL1,P_PNL1,LOC1
"""
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,VN1,SKU3,SKU3,ACC2,P_ACC2,CH2,P_CH2,REG2,P_REG2,DD2,P_DD2,PNL2,P_PNL2,LOC2,P_LOC2
"""
    date_csv = """Version,Version Name,Transition Start Date
V1,VN1,2023-01-01
"""
    inputs = {
        "ItemMaster": pd.read_csv(StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(StringIO(assortment_sell_out_csv)),
        "Date": pd.read_csv(StringIO(date_csv)),
    }

    outputs = main(inputs)

    # Output_AssortmentFinal should only contain the new SKU from the first transition group (SKU2).
    # It should not contain anything from the second group (SKU3, SKU4).
    output_final = outputs["AssortmentFinal"]
    assert len(output_final) == 1
    assert output_final["Item"].iloc[0] == "SKU2"
    assert output_final["Planning Account"].iloc[0] == "P_ACC1"
    assert output_final["Planning Region"].iloc[0] == "P_REG1"
    assert output_final["Planning Demand Domain"].iloc[0] == "SKU2-P_REG1"

    # Symmetrically, Output_AssortmentSellOut should only contain the new SKU from the second group (SKU4).
    output_sellout = outputs["AssortmentSellOut"]
    assert len(output_sellout) == 1
    assert output_sellout["Planning Item"].iloc[0] == "SKU4"
    assert output_sellout["Planning Account"].iloc[0] == "P_ACC2"
    assert output_sellout["Planning Region"].iloc[0] == "P_REG2"
    assert output_sellout["Planning Demand Domain"].iloc[0] == "SKU4-P_REG2"


def test_transition_flag_union_and_deduplication():
    """
    Validates that 'Output_TransitionFlag' correctly unions all rows from both expanded
    assortment tables and properly removes duplicate rows.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU1,T_ITEM1,P_SKU1
SKU2,T_ITEM1,P_SKU2
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,VN1,SKU1,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,PNL1,PPNL1,PLOC1
"""
    # This row is identical to the AssortmentFinal row in terms of planning dimensions,
    # which should create a duplicate after expansion.
    # The second row has a different Planning Account, so it should create unique rows.
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,VN1,SKU1,SKU1,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,PNL1,PPNL1,L1,PLOC1
V1,VN1,SKU1,SKU1,A2,PA2,C1,PC1,R1,PR1,DD1,PDD1,PNL1,PPNL1,L1,PLOC1
"""
    date_csv = """Version,Version Name,Transition Start Date
V1,VN1,2023-05-05
"""
    inputs = {
        "ItemMaster": pd.read_csv(StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(StringIO(assortment_sell_out_csv)),
        "Date": pd.read_csv(StringIO(date_csv)),
    }

    outputs = main(inputs)

    # AssortmentFinal expands to (SKU1, PA1, PC1, PR1, PPNL1, PLOC1) and (SKU2, PA1, ...).
    # AssortmentSellOut expands to:
    # 1. (SKU1, PA1, PC1, PR1, PPNL1, PLOC1) and (SKU2, PA1, ...) -> duplicates of the above
    # 2. (SKU1, PA2, PC1, PR1, PPNL1, PLOC1) and (SKU2, PA2, ...) -> unique rows
    # Total unique rows in TransitionFlag should be 4.
    output_transition_flag = outputs["TransitionFlag"]
    assert len(output_transition_flag) == 4

    # Check for presence of the 4 unique combinations
    unique_rows = output_transition_flag[["Planning Item", "Planning Account"]].drop_duplicates()
    assert len(unique_rows) == 4
    assert all(output_transition_flag["Partial Week"] == "2023-05-05")
    assert all(output_transition_flag["Transition Flag"] == 1)


def test_empty_assortment_final_input():
    """
    Tests the behavior when the 'AssortmentFinal' input is empty. Expects
    'Output_AssortmentFinal' to be empty, while other outputs are generated based on
    'AssortmentSellOut'.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU10,T_ITEM10,P_SKU10
SKU20,T_ITEM10,P_SKU20
"""
    # Empty AssortmentFinal input
    assortment_final_csv = "Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location\n"
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,VN1,SKU10,SKU10,ACC1,P_ACC1,CH1,P_CH1,REG1,P_REG1,DD1,P_DD1,PNL1,P_PNL1,LOC1,P_LOC1
"""
    date_csv = """Version,Version Name,Transition Start Date
V1,VN1,2023-10-10
"""
    inputs = {
        "ItemMaster": pd.read_csv(StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(StringIO(assortment_sell_out_csv)),
        "Date": pd.read_csv(StringIO(date_csv)),
    }

    outputs = main(inputs)

    # Output_AssortmentFinal should be empty
    assert outputs["AssortmentFinal"].empty

    # Output_AssortmentSellOut should have one new row for SKU20
    output_sellout = outputs["AssortmentSellOut"]
    assert len(output_sellout) == 1
    assert output_sellout["Planning Item"].iloc[0] == "SKU20"
    assert output_sellout["Planning Demand Domain"].iloc[0] == "SKU20-P_REG1"

    # Output_TransitionFlag should have two rows (SKU10, SKU20) from the SellOut expansion
    output_transition_flag = outputs["TransitionFlag"]
    assert len(output_transition_flag) == 2
    items = sorted(output_transition_flag["Planning Item"].tolist())
    assert items == ["SKU10", "SKU20"]
    assert all(output_transition_flag["Partial Week"] == "2023-10-10")


def test_empty_assortment_sellout_input():
    """
    Tests the behavior when the 'AssortmentSellOut' input is empty. Expects
    'Output_AssortmentSellOut' to be empty, while other outputs are generated based on
    'AssortmentFinal'.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU10,T_ITEM10,P_SKU10
SKU20,T_ITEM10,P_SKU20
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,VN1,SKU10,ACC1,P_ACC1,CH1,P_CH1,REG1,P_REG1,DD1,P_DD1,PNL1,P_PNL1,LOC1
"""
    # Empty AssortmentSellOut input
    assortment_sell_out_csv = "Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location\n"
    date_csv = """Version,Version Name,Transition Start Date
V1,VN1,2023-11-11
"""
    inputs = {
        "ItemMaster": pd.read_csv(StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(StringIO(assortment_sell_out_csv)),
        "Date": pd.read_csv(StringIO(date_csv)),
    }

    outputs = main(inputs)

    # Output_AssortmentSellOut should be empty
    assert outputs["AssortmentSellOut"].empty

    # Output_AssortmentFinal should have one new row for SKU20
    output_final = outputs["AssortmentFinal"]
    assert len(output_final) == 1
    assert output_final["Item"].iloc[0] == "SKU20"
    assert output_final["Planning Demand Domain"].iloc[0] == "SKU20-P_REG1"

    # Output_TransitionFlag should have two rows (SKU10, SKU20) from the Final expansion
    output_transition_flag = outputs["TransitionFlag"]
    assert len(output_transition_flag) == 2
    items = sorted(output_transition_flag["Planning Item"].tolist())
    assert items == ["SKU10", "SKU20"]
    assert all(output_transition_flag["Partial Week"] == "2023-11-11")

def test_empty_item_master_input():
    """
    Tests the scenario where 'ItemMaster' is empty. Expects no new rows to be generated, 
    resulting in empty 'Output_AssortmentFinal' and 'Output_AssortmentSellOut'.
    """
    item_master_csv = "Item,Transition Item,Planning Item"
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,I1,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,I1,PI1,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1,PL1
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

    outputs = main(inputs)

    assert outputs["AssortmentFinal"].empty, "AssortmentFinal should be empty when ItemMaster is empty"
    assert outputs["AssortmentSellOut"].empty, "AssortmentSellOut should be empty when ItemMaster is empty"
    assert outputs["TransitionFlag"].empty, "TransitionFlag should be empty when ItemMaster is empty"


def test_item_not_in_item_master():
    """
    Tests when an 'Item' or 'Planning Item' from an assortment input does not have a 
    corresponding entry in 'ItemMaster'. Expects this row to be ignored and no expansion to occur for it.
    """
    item_master_csv = """Item,Transition Item,Planning Item
I1,T1,P1
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,I2,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,I1,I3,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1,PL1
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

    outputs = main(inputs)

    assert outputs["AssortmentFinal"].empty, "AssortmentFinal should be empty for items not in ItemMaster"
    assert outputs["AssortmentSellOut"].empty, "AssortmentSellOut should be empty for items not in ItemMaster"
    assert outputs["TransitionFlag"].empty, "TransitionFlag should be empty when no items are expanded"


def test_single_item_per_transition_group():
    """
    Tests a 'Transition Item' that maps to only one 'Item'. Expects no new rows to be 
    generated as there are no other SKUs to expand to.
    """
    item_master_csv = """Item,Transition Item,Planning Item
I1,T1,P1
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,I1,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,I1,I1,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1,PL1
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

    outputs = main(inputs)

    assert outputs["AssortmentFinal"].empty, "AssortmentFinal should be empty for single-item transition groups"
    assert outputs["AssortmentSellOut"].empty, "AssortmentSellOut should be empty for single-item transition groups"


def test_all_skus_already_in_assortment():
    """
    Tests a scenario where an assortment already contains all SKUs for a given transition group. 
    The anti-join should correctly result in zero new rows for that group in the final outputs.
    """
    item_master_csv = """Item,Transition Item,Planning Item
I1,T1,P1
I2,T1,P2
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,I1,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1
1,V1,I2,A1,PA1,C1,PC1,R1,PR1,DD2,PDD2,P1,PP1,L1
"""
    assortment_sellout_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,I1,I1,A1,PA1,C1,PC1,R1,PR1,DD1,PDD1,P1,PP1,L1,PL1
1,V1,I2,I2,A1,PA1,C1,PC1,R1,PR1,DD2,PDD2,P1,PP1,L1,PL1
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

    outputs = main(inputs)

    assert outputs["AssortmentFinal"].empty, "AssortmentFinal should be empty when all SKUs are already in the assortment"
    assert outputs["AssortmentSellOut"].empty, "AssortmentSellOut should be empty when all SKUs are already in the assortment"

def test_duplicate_input_assortment_rows():
    """
    Tests with fully duplicate rows in the input assortment tables.
    Verifies the expansion logic handles this correctly and the 'TransitionFlag' output deduplicates the final result.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU1,BASE1,SKU1
SKU2,BASE1,SKU2
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,Actuals,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1
V1,Actuals,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1
"""
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,Actuals,SKU1,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1,P_L1
V1,Actuals,SKU1,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1,P_L1
"""
    date_csv = """Version,Version Name,Transition Start Date
V1,Actuals,2023-01-01
"""
    item_master_df = pd.read_csv(StringIO(item_master_csv))
    assortment_final_df = pd.read_csv(StringIO(assortment_final_csv))
    assortment_sell_out_df = pd.read_csv(StringIO(assortment_sell_out_csv))
    date_df = pd.read_csv(StringIO(date_csv), parse_dates=['Transition Start Date'])

    outputs = main({
        "ItemMaster": item_master_df,
        "AssortmentFinal": assortment_final_df,
        "AssortmentSellOut": assortment_sell_out_df,
        "Date": date_df
    })
    
    output_transition_flag_df = outputs['TransitionFlag']

    expected_transition_flag_data = [
        {'Version Name': 'Actuals', 'Planning Item': 'SKU1', 'Planning Account': 'P_ACC1', 'Planning Channel': 'P_CH1', 'Planning Region': 'P_R1', 'Planning Demand Domain': 'SKU1-P_R1', 'Planning PnL': 'P_PNL1', 'Planning Location': 'P_L1', 'Partial Week': pd.Timestamp('2023-01-01'), 'Transition Flag': 1},
        {'Version Name': 'Actuals', 'Planning Item': 'SKU2', 'Planning Account': 'P_ACC1', 'Planning Channel': 'P_CH1', 'Planning Region': 'P_R1', 'Planning Demand Domain': 'SKU2-P_R1', 'Planning PnL': 'P_PNL1', 'Planning Location': 'P_L1', 'Partial Week': pd.Timestamp('2023-01-01'), 'Transition Flag': 1}
    ]
    
    expected_df = pd.DataFrame(expected_transition_flag_data)
    
    # Sort both for consistent comparison
    sort_cols = sorted(expected_df.columns.tolist())
    actual_sorted = output_transition_flag_df.sort_values(by=sort_cols).reset_index(drop=True)
    expected_sorted = expected_df.sort_values(by=sort_cols).reset_index(drop=True)

    assert expected_sorted.equals(actual_sorted), "TransitionFlag output should be deduplicated despite duplicate inputs."


def test_empty_date_table():
    """
    Tests the behavior when the 'Date' input table is empty.
    Expects the 'Partial Week' column in 'Output_TransitionFlag' to be null or empty.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU1,BASE1,SKU1
SKU2,BASE1,SKU2
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,Actuals,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1
"""
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,Actuals,SKU1,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1,P_L1
"""
    date_csv = "Version,Version Name,Transition Start Date\n"
    
    item_master_df = pd.read_csv(StringIO(item_master_csv))
    assortment_final_df = pd.read_csv(StringIO(assortment_final_csv))
    assortment_sell_out_df = pd.read_csv(StringIO(assortment_sell_out_csv))
    date_df = pd.read_csv(StringIO(date_csv), parse_dates=['Transition Start Date'])

    outputs = main({
        "ItemMaster": item_master_df,
        "AssortmentFinal": assortment_final_df,
        "AssortmentSellOut": assortment_sell_out_df,
        "Date": date_df
    })
    
    output_transition_flag_df = outputs['TransitionFlag']

    expected_transition_flag_data = [
        {'Version Name': 'Actuals', 'Planning Item': 'SKU1', 'Planning Account': 'P_ACC1', 'Planning Channel': 'P_CH1', 'Planning Region': 'P_R1', 'Planning Demand Domain': 'SKU1-P_R1', 'Planning PnL': 'P_PNL1', 'Planning Location': 'P_L1', 'Partial Week': pd.NaT, 'Transition Flag': 1},
        {'Version Name': 'Actuals', 'Planning Item': 'SKU2', 'Planning Account': 'P_ACC1', 'Planning Channel': 'P_CH1', 'Planning Region': 'P_R1', 'Planning Demand Domain': 'SKU2-P_R1', 'Planning PnL': 'P_PNL1', 'Planning Location': 'P_L1', 'Partial Week': pd.NaT, 'Transition Flag': 1}
    ]
    
    expected_df = pd.DataFrame(expected_transition_flag_data)
    expected_df['Partial Week'] = pd.to_datetime(expected_df['Partial Week'])

    sort_cols = sorted(expected_df.columns.tolist())
    actual_sorted = output_transition_flag_df.sort_values(by=sort_cols).reset_index(drop=True)
    expected_sorted = expected_df.sort_values(by=sort_cols).reset_index(drop=True)
    
    assert actual_sorted.equals(expected_sorted), "Partial Week column should contain nulls when Date table is empty."


def test_null_values_in_join_keys():
    """
    Tests with null values in key columns used for joins (e.g., 'Item', 'Planning Region').
    Expects rows with null keys to be excluded from the expansion logic.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU1,BASE1,SKU1
SKU2,BASE1,SKU2
SKU3,BASE2,SKU3
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,Actuals,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1
V1,Actuals,,ACC2,P_ACC2,CH2,P_CH2,R2,P_R2,DD2,P_DD2,PNL2,P_PNL2,L2
V1,Actuals,SKU3,ACC3,P_ACC3,CH3,P_CH3,R3,,DD3,P_DD3,PNL3,P_PNL3,L3
"""
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,Actuals,SKU1,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1,P_L1
"""
    date_csv = """Version,Version Name,Transition Start Date
V1,Actuals,2023-01-01
"""
    item_master_df = pd.read_csv(StringIO(item_master_csv))
    assortment_final_df = pd.read_csv(StringIO(assortment_final_csv))
    assortment_sell_out_df = pd.read_csv(StringIO(assortment_sell_out_csv))
    date_df = pd.read_csv(StringIO(date_csv))

    outputs = main({
        "ItemMaster": item_master_df,
        "AssortmentFinal": assortment_final_df,
        "AssortmentSellOut": assortment_sell_out_df,
        "Date": date_df
    })
    
    output_assortment_final_df = outputs['AssortmentFinal']

    expected_assortment_final_data = [
        {'Version Name': 'Actuals', 'Item': 'SKU2', 'Planning Account': 'P_ACC1', 'Planning Channel': 'P_CH1', 'Planning Region': 'P_R1', 'Planning Demand Domain': 'SKU2-P_R1', 'Planning PnL': 'P_PNL1', 'Location': 'L1', 'Assortment Final': 1, 'Transition Sell In Assortment': 1}
    ]
    
    expected_df = pd.DataFrame(expected_assortment_final_data)

    assert len(output_assortment_final_df) == 1, "Only one new row should be generated from valid inputs."
    
    sort_cols = sorted(expected_df.columns.tolist())
    actual_sorted = output_assortment_final_df.sort_values(by=sort_cols).reset_index(drop=True)
    expected_sorted = expected_df.sort_values(by=sort_cols).reset_index(drop=True)
    
    assert actual_sorted.equals(expected_sorted), "Rows with null join keys should be excluded from expansion."


def test_multiple_assortment_lines_for_same_item():
    """
    Tests when the same 'Item' exists in 'AssortmentFinal' multiple times but with different planning dimensions (e.g., different Planning Region).
    Verifies that expansion occurs correctly for each unique line.
    """
    item_master_csv = """Item,Transition Item,Planning Item
SKU1,BASE1,SKU1
SKU2,BASE1,SKU2
"""
    assortment_final_csv = """Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
V1,Actuals,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1
V1,Actuals,SKU1,ACC2,P_ACC2,CH2,P_CH2,R2,P_R2,DD2,P_DD2,PNL2,P_PNL2,L2
"""
    assortment_sell_out_csv = """Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
V1,Actuals,SKU1,SKU1,ACC1,P_ACC1,CH1,P_CH1,R1,P_R1,DD1,P_DD1,PNL1,P_PNL1,L1,P_L1
"""
    date_csv = """Version,Version Name,Transition Start Date
V1,Actuals,2023-01-01
"""
    item_master_df = pd.read_csv(StringIO(item_master_csv))
    assortment_final_df = pd.read_csv(StringIO(assortment_final_csv))
    assortment_sell_out_df = pd.read_csv(StringIO(assortment_sell_out_csv))
    date_df = pd.read_csv(StringIO(date_csv))

    outputs = main({
        "ItemMaster": item_master_df,
        "AssortmentFinal": assortment_final_df,
        "AssortmentSellOut": assortment_sell_out_df,
        "Date": date_df
    })
    
    output_assortment_final_df = outputs['AssortmentFinal']

    expected_assortment_final_data = [
        {'Version Name': 'Actuals', 'Item': 'SKU2', 'Planning Account': 'P_ACC1', 'Planning Channel': 'P_CH1', 'Planning Region': 'P_R1', 'Planning Demand Domain': 'SKU2-P_R1', 'Planning PnL': 'P_PNL1', 'Location': 'L1', 'Assortment Final': 1, 'Transition Sell In Assortment': 1},
        {'Version Name': 'Actuals', 'Item': 'SKU2', 'Planning Account': 'P_ACC2', 'Planning Channel': 'P_CH2', 'Planning Region': 'P_R2', 'Planning Demand Domain': 'SKU2-P_R2', 'Planning PnL': 'P_PNL2', 'Location': 'L2', 'Assortment Final': 1, 'Transition Sell In Assortment': 1}
    ]
    
    expected_df = pd.DataFrame(expected_assortment_final_data)

    sort_cols = sorted(expected_df.columns.tolist())
    actual_sorted = output_assortment_final_df.sort_values(by=sort_cols).reset_index(drop=True)
    expected_sorted = expected_df.sort_values(by=sort_cols).reset_index(drop=True)
    
    assert actual_sorted.equals(expected_sorted), "Expansion should occur for each unique input line, even with the same item."

def test_overlapping_rows_from_final_and_sellout_expansion():
    """
    Tests a scenario where the expansion of 'AssortmentFinal' and 'AssortmentSellOut' create identical rows.
    Verifies the 'Union' and 'deduplicate' step for 'Output_TransitionFlag' produces a single, unique set of rows,
    correctly handling the overlap.
    """
    item_master_csv = """Item,Transition Item
SKU_A,BC_1
SKU_B,BC_1
"""
    assortment_final_csv = """Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location
V1,SKU_A,PA1,PC1,PR1,PPNL1,PLOC1
"""
    assortment_sellout_csv = """Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location
V1,SKU_A,PA1,PC1,PR1,PPNL1,PLOC1
"""
    date_csv = """Transition Start Date
2023-01-01
"""
    # Step 4: AssortmentFinal_Expanded will contain rows for SKU_A and SKU_B.
    # Step 5: AssortmentSellOut_Expanded will also contain rows for SKU_A and SKU_B.
    # Step 8: The union of these two identical sets should be deduplicated, resulting
    # in exactly two rows (one for SKU_A, one for SKU_B) in the final TransitionFlag output.
    # A simple concatenation would yield 4 rows, so a result of 2 proves deduplication.

    inputs = {
        "ItemMaster": pd.read_csv(StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(StringIO(date_csv)),
    }

    result = main(inputs)
    output_df = result["TransitionFlag"]

    assert len(output_df) == 2, "TransitionFlag should contain 2 unique rows after deduplication"
    # Verify the contents to be sure
    expected_items = {'SKU_A', 'SKU_B'}
    actual_items = set(output_df['Planning Item'].unique())
    assert actual_items == expected_items, "TransitionFlag should contain one row for each item in the basecode group"


def test_inconsistent_item_master_mapping():
    """
    Tests the behavior when an 'Item' is mapped to more than one 'Transition Item' in 'ItemMaster'.
    The test validates that the system's behavior is predictable (e.g., last match wins).
    """
    item_master_csv = """Item,Transition Item
SKU_TARGET,BC_2
SKU_1,BC_1
SKU_X,BC_1
SKU_1,BC_2
"""
    assortment_final_csv = """Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location
V1,SKU_1,PA1,PC1,PR1,PPNL1,PLOC1
"""
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\n"  # Empty
    date_csv = """Transition Start Date
2023-01-01
"""
    # Logic: The Item-to-Basecode map should have SKU_1 -> BC_2 (last match wins).
    # The Basecode-to-SKU map will have BC_2 -> [SKU_TARGET, SKU_1].
    # Expansion of the AssortmentFinal row for SKU_1 will generate rows for SKU_TARGET and SKU_1.
    # The anti-join for Output_AssortmentFinal should filter out the original SKU_1 row,
    # leaving only the newly generated SKU_TARGET row.

    inputs = {
        "ItemMaster": pd.read_csv(StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(StringIO(date_csv)),
    }

    result = main(inputs)
    output_df = result["AssortmentFinal"]

    assert len(output_df) == 1
    assert output_df["Item"].iloc[0] == "SKU_TARGET", "Expansion should use the last matching Transition Item (BC_2)"


def test_anti_join_logic_precision():
    """
    Specifically validates that the left anti-join correctly identifies ONLY newly generated rows,
    excluding all rows that were present in the original input tables.
    """
    item_master_csv = """Item,Transition Item
SKU_A,BC_1
SKU_B,BC_1
"""
    # Input contains all items for the basecode, so no new rows should be generated.
    assortment_final_csv = """Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location
V1,SKU_A,PA1,PC1,PR1,PPNL1,PLOC1
V1,SKU_B,PA1,PC1,PR1,PPNL1,PLOC1
"""
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\n"  # Empty
    date_csv = """Transition Start Date
2023-01-01
"""
    # Logic: The expansion of the SKU_A row will generate rows for SKU_A and SKU_B.
    # Both of these generated rows already exist in the original AssortmentFinal input.
    # Therefore, the left anti-join should result in an empty DataFrame.

    inputs = {
        "ItemMaster": pd.read_csv(StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(StringIO(date_csv)),
    }

    result = main(inputs)
    output_df = result["AssortmentFinal"]

    assert output_df.empty, "Output should be empty as all expanded rows existed in the original input"


def test_column_mapping_for_transition_flag_union():
    """
    Verifies that the column mapping from 'AssortmentFinal_Expanded' ('Item' to 'Planning Item',
    'Location' to 'Planning Location') is performed correctly before the union for the 'TransitionFlag' output.
    """
    item_master_csv = """Item,Transition Item
SKU_A,BC_1
SKU_B,BC_1
"""
    assortment_final_csv = """Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location
V1,SKU_A,PA1,PC1,PR1,PPNL1,FINAL_LOC
"""
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\n"  # Empty
    date_csv = """Transition Start Date
2023-01-01
"""
    # Logic:
    # 1. AssortmentFinal_Expanded will have a row for SKU_A and SKU_B.
    #    The row for SKU_A will have Item='SKU_A' and Location='FINAL_LOC'.
    # 2. When creating TransitionFlag, these rows are processed.
    # 3. The `Item` column must be mapped to `Planning Item`.
    # 4. The `Location` column must be mapped to `Planning Location`.
    # We will check the row for SKU_A to ensure its Location was mapped correctly.

    inputs = {
        "ItemMaster": pd.read_csv(StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(StringIO(date_csv)),
    }

    result = main(inputs)
    output_df = result["TransitionFlag"]

    # Filter for one of the rows to check its specific values
    test_row = output_df[output_df["Planning Item"] == "SKU_A"]

    assert not test_row.empty, "Expected row for SKU_A not found in TransitionFlag output"
    assert test_row["Planning Location"].iloc[0] == "FINAL_LOC", "Location from AssortmentFinal was not correctly mapped to Planning Location"
    assert test_row["Planning Account"].iloc[0] == "PA1", "Dimension column was not correctly carried over"
    assert "Item" not in output_df.columns, "Original 'Item' column should not exist in TransitionFlag"
    assert "Location" not in output_df.columns, "Original 'Location' column should not exist in TransitionFlag"