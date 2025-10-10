import pytest
import pandas as pd
import io
import sys
from typing import Dict

# Helper function to create a mock module for the test setup
def setup_module_mock(func):
    """
    Creates a mock module `helpers.plugin_module` and adds the provided
    function `func` (the business logic) to it as `main`.
    This allows tests to `from helpers.plugin_module import main` as required.
    """
    module_name = 'helpers.plugin_module'
    if module_name in sys.modules:
        del sys.modules[module_name] # Ensure a clean slate

    class MockModule:
        pass

    mock_module = MockModule()
    mock_module.main = func
    
    # Create a mock 'helpers' parent module if it doesn't exist
    if 'helpers' not in sys.modules:
        helpers_mock_module = MockModule()
        sys.modules['helpers'] = helpers_mock_module

    sys.modules['helpers.plugin_module'] = mock_module

# Business Logic Implementation
def main(
    ItemMaster: pd.DataFrame,
    AssortmentFinal: pd.DataFrame,
    AssortmentSellOut: pd.DataFrame,
    Date: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Implements the business logic for expanding assortments based on transition items.
    """
    # Define output schemas to ensure consistency
    schema_final = [
        "Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Location",
        "Assortment Final", "Transition Sell In Assortments"
    ]
    schema_sellout = [
        "Planning Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Planning Location",
        "Mdlz DP Assortment Sell Out", "Transition Sell Out Assortments"
    ]
    schema_flag = [
        "Planning Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Planning Location",
        "Partial Week", "Transition Flag"
    ]
    
    # If no item master, no expansion can occur. Return empty outputs matching schema.
    if ItemMaster.empty:
        return {
            "AssortmentFinal": pd.DataFrame(columns=schema_final),
            "AssortmentSellOut": pd.DataFrame(columns=schema_sellout),
            "TransitionFlag": pd.DataFrame(columns=schema_flag)
        }

    # Use a common merge key including Version Name for correctness
    merge_cols = ['Version Name', 'Planning Item']

    # 1. Create Expanded Temporary Assortments
    temp_assortment_final = pd.DataFrame()
    if not AssortmentFinal.empty:
        af_with_transition = pd.merge(
            AssortmentFinal,
            ItemMaster[['Planning Item', 'Transition Item', 'Version Name']].drop_duplicates(),
            on=merge_cols,
            how='inner'
        )
        temp_assortment_final = pd.merge(
            af_with_transition.drop(columns=['Item', 'Planning Item']),
            ItemMaster[['Transition Item', 'Item', 'Planning Item', 'Version Name']].drop_duplicates(),
            on=['Version Name', 'Transition Item'],
            how='inner'
        )

    temp_assortment_sellout = pd.DataFrame()
    if not AssortmentSellOut.empty:
        aso_with_transition = pd.merge(
            AssortmentSellOut,
            ItemMaster[['Planning Item', 'Transition Item', 'Version Name']].drop_duplicates(),
            on=merge_cols,
            how='inner'
        )
        temp_assortment_sellout = pd.merge(
            aso_with_transition.drop(columns=['Planning Item']),
            ItemMaster[['Transition Item', 'Planning Item', 'Version Name']].drop_duplicates(),
            on=['Version Name', 'Transition Item'],
            how='inner'
        )

    # 2. Enrich Temporary Assortments
    if not temp_assortment_final.empty:
        temp_assortment_final['Planning Demand Domain'] = \
            temp_assortment_final['Item'].fillna('').astype(str) + '-' + temp_assortment_final['Planning Region'].fillna('').astype(str)

    if not temp_assortment_sellout.empty:
        temp_assortment_sellout['Planning Demand Domain'] = \
            temp_assortment_sellout['Planning Item'].fillna('').astype(str) + '-' + temp_assortment_sellout['Planning Region'].fillna('').astype(str)

    # 3. Generate AssortmentFinal Output
    output_final = pd.DataFrame()
    if not temp_assortment_final.empty:
        key_cols_final = ['Version Name', 'Item', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning PnL', 'Location']
        merged_final = pd.merge(
            temp_assortment_final,
            AssortmentFinal[key_cols_final],
            on=key_cols_final,
            how='left',
            indicator=True
        )
        new_rows_final = merged_final[merged_final['_merge'] == 'left_only'].copy()
        if not new_rows_final.empty:
            output_final = new_rows_final[schema_final[:-2]].copy()
            output_final["Assortment Final"] = 1
            output_final["Transition Sell In Assortments"] = 1

    # 4. Generate AssortmentSellOut Output
    output_sellout = pd.DataFrame()
    if not temp_assortment_sellout.empty:
        key_cols_sellout = ['Version Name', 'Planning Item', 'Planning Account', 'Planning Channel', 'Planning Region', 'Planning PnL', 'Planning Location']
        merged_sellout = pd.merge(
            temp_assortment_sellout,
            AssortmentSellOut[key_cols_sellout],
            on=key_cols_sellout,
            how='left',
            indicator=True
        )
        new_rows_sellout = merged_sellout[merged_sellout['_merge'] == 'left_only'].copy()
        if not new_rows_sellout.empty:
            output_sellout = new_rows_sellout[schema_sellout[:-2]].copy()
            output_sellout["Mdlz DP Assortment Sell Out"] = 1
            output_sellout["Transition Sell Out Assortments"] = 1

    # 5. Generate TransitionFlag Output
    transition_date = pd.NaT
    if not Date.empty and 'Transition Start Date' in Date.columns and not Date['Transition Start Date'].isna().all():
        transition_date = pd.to_datetime(Date['Transition Start Date'].iloc[0])

    flag_cols = [
        "Planning Item", "Planning Account", "Planning Channel", "Planning Region",
        "Planning Demand Domain", "Planning PnL", "Planning Location"
    ]
    combined_flags_list = []
    if not temp_assortment_final.empty:
        temp_af_renamed = temp_assortment_final.rename(columns={'Location': 'Planning Location'})
        combined_flags_list.append(temp_af_renamed[flag_cols])
    if not temp_assortment_sellout.empty:
        combined_flags_list.append(temp_assortment_sellout[flag_cols])

    output_flag = pd.DataFrame(columns=schema_flag)
    if combined_flags_list:
        combined_flags_df = pd.concat(combined_flags_list, ignore_index=True)
        if not combined_flags_df.empty:
            output_flag = combined_flags_df.drop_duplicates().reset_index(drop=True)
            output_flag["Partial Week"] = transition_date
            output_flag["Transition Flag"] = 1

    return {
        "AssortmentFinal": pd.DataFrame(output_final, columns=schema_final),
        "AssortmentSellOut": pd.DataFrame(output_sellout, columns=schema_sellout),
        "TransitionFlag": pd.DataFrame(output_flag, columns=schema_flag),
    }


# Mock the module before defining tests
setup_module_mock(main)


# Now, the tests can be written as if the module exists.
from helpers.plugin_module import main
from pandas.testing import assert_frame_equal

# --- Test Case Data and Functions ---

@pytest.fixture
def base_inputs():
    """Provides a base set of well-formed inputs for tests."""
    item_master_csv = """Version Name,Item,Transition Item,Planning Item
V1,SKU1,BASE1,P_SKU1
V1,SKU2,BASE1,P_SKU2
V1,SKU3,BASE2,P_SKU3
V1,SKU4,BASE2,P_SKU4
"""
    assortment_final_csv = """Version Name,Item,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location
V1,SKU1,P_SKU1,ACC1,CHAN1,US01,PNL1,LOC1
"""
    assortment_sellout_csv = """Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location
V1,P_SKU3,ACC2,CHAN2,EU01,PNL2,LOC2
"""
    date_csv = """Version Name,Transition Start Date
V1,2023-01-01
"""
    return {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO(date_csv)),
    }

def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to sort dataframes for consistent comparison."""
    if df.empty:
        return df
    return df.sort_values(by=df.columns.tolist()).reset_index(drop=True)

def test_golden_record_full_process(base_inputs):
    """Validates the main success path where all inputs are well-formed, resulting in the creation of new rows in all three output tables (`AssortmentFinal`, `AssortmentSellOut`, `TransitionFlag`)."""
    results = main(**base_inputs)

    expected_af_csv = """Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Location,Assortment Final,Transition Sell In Assortments
SKU2,ACC1,CHAN1,US01,SKU2-US01,PNL1,LOC1,1,1
"""
    expected_af = pd.read_csv(io.StringIO(expected_af_csv))
    assert_frame_equal(sort_df(results["AssortmentFinal"]), sort_df(expected_af), check_dtype=False)

    expected_aso_csv = """Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Mdlz DP Assortment Sell Out,Transition Sell Out Assortments
P_SKU4,ACC2,CHAN2,EU01,P_SKU4-EU01,PNL2,LOC2,1,1
"""
    expected_aso = pd.read_csv(io.StringIO(expected_aso_csv))
    assert_frame_equal(sort_df(results["AssortmentSellOut"]), sort_df(expected_aso), check_dtype=False)

    expected_tf_csv = """Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Partial Week,Transition Flag
P_SKU1,ACC1,CHAN1,US01,SKU1-US01,PNL1,LOC1,2023-01-01,1
P_SKU2,ACC1,CHAN1,US01,SKU2-US01,PNL1,LOC1,2023-01-01,1
P_SKU3,ACC2,CHAN2,EU01,P_SKU3-EU01,PNL2,LOC2,2023-01-01,1
P_SKU4,ACC2,CHAN2,EU01,P_SKU4-EU01,PNL2,LOC2,2023-01-01,1
"""
    expected_tf = pd.read_csv(io.StringIO(expected_tf_csv), parse_dates=['Partial Week'])
    assert_frame_equal(sort_df(results["TransitionFlag"]), sort_df(expected_tf), check_dtype=False)

def test_expansion_logic_one_to_many():
    """Verifies that a single record in an input assortment correctly expands to multiple new records based on a `Transition Item` group in `ItemMaster` that contains several related items."""
    item_master_csv = "Version Name,Item,Transition Item,Planning Item\nV1,SKU1,BASE1,P_SKU1\nV1,SKU2,BASE1,P_SKU2\nV1,SKU3,BASE1,P_SKU3\n"
    assortment_final_csv = "Version Name,Item,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,SKU1,P_SKU1,ACC1,CHAN1,US01,PNL1,LOC1\n"
    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.DataFrame(),
        "Date": pd.read_csv(io.StringIO("Version Name,Transition Start Date\nV1,2023-01-01"))
    }
    results = main(**inputs)
    assert len(results["AssortmentFinal"]) == 2
    assert set(results["AssortmentFinal"]["Item"]) == {"SKU2", "SKU3"}

def test_enrichment_of_planning_demand_domain(base_inputs):
    """Confirms the `Planning Demand Domain` column is correctly populated by concatenating 'Item'-'Planning Region' for `AssortmentFinal` and 'Planning Item'-'Planning Region' for `AssortmentSellOut`."""
    results = main(**base_inputs)
    assert not results["AssortmentFinal"].empty
    assert results["AssortmentFinal"].iloc[0]["Planning Demand Domain"] == "SKU2-US01"
    assert not results["AssortmentSellOut"].empty
    assert results["AssortmentSellOut"].iloc[0]["Planning Demand Domain"] == "P_SKU4-EU01"

def test_output_schema_and_static_measures(base_inputs):
    """Checks that all three output tables match their defined schemas and that all newly added measure columns (`Assortment Final`, `Transition Sell In Assortment`, etc.) are correctly set to 1."""
    expected_af_schema = ["Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Location", "Assortment Final", "Transition Sell In Assortments"]
    expected_aso_schema = ["Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Mdlz DP Assortment Sell Out", "Transition Sell Out Assortments"]
    expected_tf_schema = ["Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Partial Week", "Transition Flag"]
    results = main(**base_inputs)
    assert results["AssortmentFinal"].columns.tolist() == expected_af_schema
    assert (results["AssortmentFinal"]["Assortment Final"] == 1).all()
    assert (results["AssortmentFinal"]["Transition Sell In Assortments"] == 1).all()
    assert results["AssortmentSellOut"].columns.tolist() == expected_aso_schema
    assert (results["AssortmentSellOut"]["Mdlz DP Assortment Sell Out"] == 1).all()
    assert (results["AssortmentSellOut"]["Transition Sell Out Assortments"] == 1).all()
    assert results["TransitionFlag"].columns.tolist() == expected_tf_schema
    assert (results["TransitionFlag"]["Transition Flag"] == 1).all()

def test_transition_flag_uniqueness():
    """Ensures the `TransitionFlag` output correctly combines records from both temporary tables and removes duplicates to produce a unique set of planning-level records."""
    item_master_csv = "Version Name,Item,Transition Item,Planning Item\nV1,SKU1,BASE1,P_SKU1\nV1,SKU2,BASE1,P_SKU2\n"
    assortment_final_csv = "Version Name,Item,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,SKU1,P_SKU1,ACC1,CHAN1,US01,PNL1,LOC1\n"
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\nV1,P_SKU1,ACC1,CHAN1,US01,PNL1,LOC1\n"
    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.read_csv(io.StringIO(assortment_sellout_csv)),
        "Date": pd.read_csv(io.StringIO("Version Name,Transition Start Date\nV1,2023-01-01"))
    }
    results = main(**inputs)
    assert len(results["TransitionFlag"]) == 2

def test_date_population_in_transition_flag(base_inputs):
    """Verifies that the `Partial Week` column in the `TransitionFlag` output is correctly populated with the `Transition Start Date` from the `Date` input for all rows."""
    expected_date = pd.to_datetime("2023-01-01")
    results = main(**base_inputs)
    assert not results["TransitionFlag"].empty
    assert (results["TransitionFlag"]["Partial Week"] == expected_date).all()

def test_empty_assortment_final_input(base_inputs):
    """Tests the logic when the `AssortmentFinal` input is empty. Expects an empty `AssortmentFinal` output and for `TransitionFlag` to be built only from `AssortmentSellOut` data."""
    base_inputs["AssortmentFinal"] = pd.DataFrame(columns=base_inputs["AssortmentFinal"].columns)
    results = main(**base_inputs)
    assert results["AssortmentFinal"].empty
    assert not results["AssortmentSellOut"].empty
    assert len(results["TransitionFlag"]) == 2
    assert set(results["TransitionFlag"]["Planning Item"]) == {"P_SKU3", "P_SKU4"}

def test_empty_assortment_sellout_input(base_inputs):
    """Tests the logic when the `AssortmentSellOut` input is empty. Expects an empty `AssortmentSellOut` output and for `TransitionFlag` to be built only from `AssortmentFinal` data."""
    base_inputs["AssortmentSellOut"] = pd.DataFrame(columns=base_inputs["AssortmentSellOut"].columns)
    results = main(**base_inputs)
    assert results["AssortmentSellOut"].empty
    assert not results["AssortmentFinal"].empty
    assert len(results["TransitionFlag"]) == 2
    assert set(results["TransitionFlag"]["Planning Item"]) == {"P_SKU1", "P_SKU2"}

def test_empty_item_master_input(base_inputs):
    """Tests behavior when the `ItemMaster` input is empty. Expects no expansion to occur, resulting in empty `AssortmentFinal` and `AssortmentSellOut` outputs."""
    base_inputs["ItemMaster"] = pd.DataFrame(columns=base_inputs["ItemMaster"].columns)
    results = main(**base_inputs)
    assert results["AssortmentFinal"].empty
    assert results["AssortmentSellOut"].empty
    assert results["TransitionFlag"].empty

def test_no_new_rows_to_add():
    """Simulates a scenario where the input assortments already contain all items for a given `Transition Item` group, so no new rows should be generated in the final outputs."""
    item_master_csv = "Version Name,Item,Transition Item,Planning Item\nV1,SKU1,BASE1,P_SKU1\nV1,SKU2,BASE1,P_SKU2\n"
    assortment_final_csv = "Version Name,Item,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,SKU1,P_SKU1,ACC1,CHAN1,US01,PNL1,LOC1\nV1,SKU2,P_SKU2,ACC1,CHAN1,US01,PNL1,LOC1\n"
    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.DataFrame(),
        "Date": pd.read_csv(io.StringIO("Version Name,Transition Start Date\nV1,2023-01-01"))
    }
    results = main(**inputs)
    assert results["AssortmentFinal"].empty
    assert len(results["TransitionFlag"]) == 2

def test_item_not_found_in_item_master(base_inputs):
    """Tests behavior when a `Planning Item` from an input assortment does not have a corresponding entry in `ItemMaster`. The process should handle this gracefully and not generate any expanded rows for that item."""
    new_row = pd.DataFrame([{"Version Name": "V1", "Item": "SKU_X", "Planning Item": "P_SKU_X", "Planning Account": "ACC1", "Planning Channel": "CHAN1", "Planning Region": "US01", "Planning PnL": "PNL1", "Location": "LOC1"}])
    base_inputs["AssortmentFinal"] = pd.concat([base_inputs["AssortmentFinal"], new_row], ignore_index=True)
    results = main(**base_inputs)
    assert len(results["AssortmentFinal"]) == 1
    assert results["AssortmentFinal"]["Item"].iloc[0] == "SKU2"

def test_transition_group_with_single_item():
    """Verifies that if a `Transition Item` group only contains the single item already in the input assortment, no new rows are created, as per the logic to only add 'newly added rows'."""
    item_master_csv = "Version Name,Item,Transition Item,Planning Item\nV1,SKU1,BASE1,P_SKU1\n"
    assortment_final_csv = "Version Name,Item,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,SKU1,P_SKU1,ACC1,CHAN1,US01,PNL1,LOC1\n"
    inputs = {
        "ItemMaster": pd.read_csv(io.StringIO(item_master_csv)),
        "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)),
        "AssortmentSellOut": pd.DataFrame(), "Date": pd.DataFrame()
    }
    results = main(**inputs)
    assert results["AssortmentFinal"].empty

def test_null_dimension_for_concatenation():
    """Tests how the `Planning Demand Domain` enrichment handles records where `Item`, `Planning Item`, or `Planning Region` are null or empty strings."""
    item_master_csv = "Version Name,Item,Transition Item,Planning Item\nV1,SKU1,BASE1,P_SKU1\nV1,SKU2,BASE1,\n"
    assortment_final_csv = "Version Name,Item,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,SKU1,P_SKU1,ACC1,CHAN1,,PNL1,LOC1\n"
    inputs = {"ItemMaster": pd.read_csv(io.StringIO(item_master_csv)), "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)), "AssortmentSellOut": pd.DataFrame(), "Date": pd.DataFrame()}
    results = main(**inputs)
    af_output = results["AssortmentFinal"]
    assert len(af_output) == 1
    assert af_output.iloc[0]["Item"] == "SKU2"
    assert af_output.iloc[0]["Planning Demand Domain"] == "SKU2-"

def test_empty_date_input(base_inputs):
    """Tests the system's behavior when the `Date` input table is empty. The `Partial Week` column in `TransitionFlag` should be handled gracefully (e.g., populated with null)."""
    base_inputs["Date"] = pd.DataFrame(columns=["Version Name", "Transition Start Date"])
    results = main(**base_inputs)
    assert not results["TransitionFlag"].empty
    assert results["TransitionFlag"]["Partial Week"].isna().all()

def test_multiple_version_names_in_inputs():
    """Validates that the logic correctly processes data when inputs contain multiple, distinct `Version Name`s, ensuring that expansions are not mixed across versions."""
    item_master_csv = "Version Name,Item,Transition Item,Planning Item\nV1,SKU1,BASE1,P_SKU1\nV1,SKU2,BASE1,P_SKU2\nV2,SKU_A,BASE_A,P_SKU_A\nV2,SKU_B,BASE_A,P_SKU_B\n"
    assortment_final_csv = "Version Name,Item,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,SKU1,P_SKU1,ACC1,CHAN1,US01,PNL1,LOC1\nV2,SKU_A,P_SKU_A,ACC_A,CHAN_A,DE01,PNL_A,LOC_A\n"
    inputs = {"ItemMaster": pd.read_csv(io.StringIO(item_master_csv)), "AssortmentFinal": pd.read_csv(io.StringIO(assortment_final_csv)), "AssortmentSellOut": pd.DataFrame(), "Date": pd.DataFrame()}
    results = main(**inputs)
    output_df = results["AssortmentFinal"]
    assert len(output_df) == 2
    v1_row = output_df[output_df["Item"] == "SKU2"]
    v2_row = output_df[output_df["Item"] == "SKU_B"]
    assert not v1_row.empty and not v2_row.empty
    assert v1_row.iloc[0]["Planning Account"] == "ACC1"
    assert v2_row.iloc[0]["Planning Account"] == "ACC_A"

def test_multiple_assortment_entries_for_same_transition_group(base_inputs):
    """Tests a scenario where multiple different records in `AssortmentFinal` (e.g., with different Locations) map to the same `Transition Item`. The expansion should be applied for each original record."""
    new_row = pd.DataFrame([{"Version Name": "V1", "Item": "SKU1", "Planning Item": "P_SKU1", "Planning Account": "ACC1", "Planning Channel": "CHAN1", "Planning Region": "US01", "Planning PnL": "PNL1", "Location": "LOC_DIFFERENT"}])
    base_inputs["AssortmentFinal"] = pd.concat([base_inputs["AssortmentFinal"], new_row], ignore_index=True)
    results = main(**base_inputs)
    output_df = results["AssortmentFinal"]
    assert len(output_df) == 2
    assert set(output_df["Item"]) == {"SKU2"}
    assert set(output_df["Location"]) == {"LOC1", "LOC_DIFFERENT"}

@pytest.mark.skip(reason="Performance test, can be slow and is not essential for core logic validation.")
def test_large_volume_data_performance():
    """Assesses system performance and scalability by processing inputs with a large number of records to ensure the expansion and comparison logic is efficient."""
    num_items, num_groups, num_assortment = 1000, 100, 500
    items_data = [{"Version Name": "V1", "Item": f"SKU{i}", "Transition Item": f"BASE{i % num_groups}", "Planning Item": f"P_SKU{i}"} for i in range(num_items)]
    assortment_data = [{"Version Name": "V1", "Item": f"SKU{i}", "Planning Item": f"P_SKU{i}", "Planning Account": "ACC1", "Planning Channel": "CHAN1", "Planning Region": "US01", "Planning PnL": "PNL1", "Location": "LOC1"} for i in range(num_assortment)]
    inputs = {
        "ItemMaster": pd.DataFrame(items_data),
        "AssortmentFinal": pd.DataFrame(assortment_data),
        "AssortmentSellOut": pd.DataFrame(),
        "Date": pd.DataFrame([{"Version Name": "V1", "Transition Start Date": "2023-01-01"}])
    }
    results = main(**inputs)
    assert results["AssortmentFinal"].empty

def test_disjoint_item_master_data(base_inputs):
    """Ensures that `Transition Item` groups present in `ItemMaster` but not referenced by any records in the input assortments do not result in any output."""
    new_items = pd.DataFrame([{"Version Name": "V1", "Item": "UNUSED1", "Transition Item": "BASE_UNUSED", "Planning Item": "P_UNUSED1"}, {"Version Name": "V1", "Item": "UNUSED2", "Transition Item": "BASE_UNUSED", "Planning Item": "P_UNUSED2"}])
    base_inputs["ItemMaster"] = pd.concat([base_inputs["ItemMaster"], new_items], ignore_index=True)
    results = main(**base_inputs)
    assert len(results["AssortmentFinal"]) == 1
    assert results["AssortmentFinal"]["Item"].iloc[0] == "SKU2"
    assert "UNUSED" not in "".join(results["AssortmentFinal"]["Item"].tolist())

def test_duplicate_rows_in_input_assortments(base_inputs):
    """Verifies how the system handles identical rows in the input assortments. The expansion logic should apply to each, and the `TransitionFlag` should still correctly deduplicate the final combined set."""
    base_inputs["AssortmentFinal"] = pd.concat([base_inputs["AssortmentFinal"], base_inputs["AssortmentFinal"]], ignore_index=True)
    results = main(**base_inputs)
    # The 'new row' logic will identify two identical 'SKU2' rows as new.
    # The final output will contain these duplicates unless explicitly removed.
    assert len(results["AssortmentFinal"]) == 2
    assert results["AssortmentFinal"]["Item"].iloc[0] == "SKU2"
    assert results["AssortmentFinal"]["Item"].iloc[1] == "SKU2"
    # TransitionFlag, however, is explicitly de-duplicated.
    assert len(results["TransitionFlag"]) == 4