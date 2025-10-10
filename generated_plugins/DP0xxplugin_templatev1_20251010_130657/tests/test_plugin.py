import pandas as pd
import io
import pytest
import sys
from types import ModuleType

# ------------------------------------------------------------------------------------
# Main Business Logic Implementation
# This function would typically be in `helpers/plugin_module.py`
# ------------------------------------------------------------------------------------

def main(
    ItemMaster: pd.DataFrame,
    AssortmentFinal: pd.DataFrame,
    AssortmentSellOut: pd.DataFrame,
    Date: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Implements the business logic to expand assortments for transitioning items.
    """
    # Define output schemas
    output_schema = {
        "AssortmentFinal": [
            "Version Name", "Item", "Planning Account", "Planning Channel",
            "Planning Region", "Planning Demand Domain", "Planning PnL",
            "Location", "Assortment Final", "Transition Sell In Assortments"
        ],
        "AssortmentSellOut": [
            "Version Name", "Planning Item", "Planning Account", "Planning Channel",
            "Planning Region", "Planning Demand Domain", "Planning PnL",
            "Planning Location", "Mdlz DP Assortment Sell Out",
            "Transition Sell Out Assortments"
        ],
        "TransitionFlag": [
            "Version Name", "Planning Item", "Planning Account", "Planning Channel",
            "Planning Region", "Planning Demand Domain", "Planning PnL",
            "Planning Location", "Partial Week", "Transition Flag"
        ],
    }
    
    # Helper to create empty dataframes with correct schema
    def create_empty_output():
        return {
            "AssortmentFinal": pd.DataFrame(columns=output_schema["AssortmentFinal"]),
            "AssortmentSellOut": pd.DataFrame(columns=output_schema["AssortmentSellOut"]),
            "TransitionFlag": pd.DataFrame(columns=output_schema["TransitionFlag"]),
        }

    # Handle empty inputs early
    if AssortmentFinal.empty or ItemMaster.empty:
        return create_empty_output()

    # 1. Identify seed items from AssortmentFinal
    seed_items = AssortmentFinal["Item"].dropna().unique()
    if len(seed_items) == 0:
        return create_empty_output()

    # 2. Find the Transition Item (base code) for seed items
    transition_items_df = ItemMaster[ItemMaster["Item"].isin(seed_items)]
    transition_items = transition_items_df["Transition Item"].dropna().unique()
    if len(transition_items) == 0:
        return create_empty_output()

    # 3. Get the complete list of all items belonging to the identified Transition Items
    all_related_items_df = ItemMaster[ItemMaster["Transition Item"].isin(transition_items)].copy()
    all_items = all_related_items_df["Item"].dropna().unique()
    all_planning_items = all_related_items_df["Planning Item"].dropna().unique()

    # Ensure all columns used in concatenation are string type to handle potential nulls
    for df, cols in [
        (all_related_items_df, ["Item", "Planning Item"]),
        (AssortmentFinal, ["Item", "Planning Region"]),
        (AssortmentSellOut, ["Planning Item", "Planning Region"])
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # 4. Create ExpandedAssortmentFinal
    dims_af = [
        "Version Name", "Planning Account", "Planning Channel", "Planning Region",
        "Planning PnL", "Location"
    ]
    if not all(c in AssortmentFinal.columns for c in dims_af):
        unique_dims_af = pd.DataFrame(columns=dims_af)
    else:
        unique_dims_af = AssortmentFinal[dims_af].drop_duplicates()
        
    expanded_af = pd.DataFrame()
    if not unique_dims_af.empty and len(all_items) > 0:
        expanded_af = unique_dims_af.assign(key=1).merge(
            pd.DataFrame({"Item": all_items, "key": 1}), on="key"
        ).drop("key", axis=1)
        expanded_af["Planning Demand Domain"] = (
            expanded_af["Item"].astype(str) + expanded_af["Planning Region"].astype(str)
        )

    # 5. Create ExpandedAssortmentSellOut
    dims_aso = [
        "Version Name", "Planning Account", "Planning Channel", "Planning Region",
        "Planning PnL", "Planning Location"
    ]
    if AssortmentSellOut.empty or not all(c in AssortmentSellOut.columns for c in dims_aso):
        unique_dims_aso = pd.DataFrame(columns=dims_aso)
    else:
        unique_dims_aso = AssortmentSellOut[dims_aso].drop_duplicates()

    expanded_aso = pd.DataFrame()
    if not unique_dims_aso.empty and len(all_planning_items) > 0:
        expanded_aso = unique_dims_aso.assign(key=1).merge(
            pd.DataFrame({"Planning Item": all_planning_items, "key": 1}), on="key"
        ).drop("key", axis=1)
        expanded_aso["Planning Demand Domain"] = (
            expanded_aso["Planning Item"].astype(str) + expanded_aso["Planning Region"].astype(str)
        )

    # 6. Generate output AssortmentFinal
    output_af = pd.DataFrame(columns=output_schema["AssortmentFinal"])
    if not expanded_af.empty:
        key_cols_af = [
            "Version Name", "Item", "Planning Account", "Planning Channel",
            "Planning Region", "Planning PnL", "Location"
        ]
        merged_af = expanded_af.merge(
            AssortmentFinal[key_cols_af].drop_duplicates(),
            on=key_cols_af, how="left", indicator=True
        )
        new_rows_af = merged_af[merged_af["_merge"] == "left_only"].copy()
        if not new_rows_af.empty:
            new_rows_af["Assortment Final"] = 1
            new_rows_af["Transition Sell In Assortments"] = 1
            output_af = new_rows_af[output_schema["AssortmentFinal"]]

    # 7. Generate output AssortmentSellOut
    output_aso = pd.DataFrame(columns=output_schema["AssortmentSellOut"])
    if not expanded_aso.empty:
        key_cols_aso = [
            "Version Name", "Planning Item", "Planning Account", "Planning Channel",
            "Planning Region", "Planning PnL", "Planning Location"
        ]
        merged_aso = expanded_aso.merge(
            AssortmentSellOut[key_cols_aso].drop_duplicates(),
            on=key_cols_aso, how="left", indicator=True
        )
        new_rows_aso = merged_aso[merged_aso["_merge"] == "left_only"].copy()
        if not new_rows_aso.empty:
            new_rows_aso["Mdlz DP Assortment Sell Out"] = 1
            new_rows_aso["Transition Sell Out Assortments"] = 1
            output_aso = new_rows_aso[output_schema["AssortmentSellOut"]]
            
    # 8. Generate output TransitionFlag
    output_tf = pd.DataFrame(columns=output_schema["TransitionFlag"])
    if not expanded_af.empty or not expanded_aso.empty:
        tf_from_af = pd.DataFrame()
        if not expanded_af.empty:
            item_map = all_related_items_df[["Item", "Planning Item"]].drop_duplicates()
            tf_from_af = expanded_af.merge(item_map, on="Item", how="inner")
            tf_from_af = tf_from_af.rename(columns={"Location": "Planning Location"})
            tf_from_af["Planning Demand Domain"] = (
                tf_from_af["Planning Item"].astype(str) + tf_from_af["Planning Region"].astype(str)
            )

        tf_cols = [
            "Version Name", "Planning Item", "Planning Account", "Planning Channel",
            "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location"
        ]
        combined_tf = pd.concat([
            tf_from_af[tf_cols] if not tf_from_af.empty else pd.DataFrame(columns=tf_cols),
            expanded_aso[tf_cols] if not expanded_aso.empty else pd.DataFrame(columns=tf_cols)
        ]).drop_duplicates().reset_index(drop=True)

        if not combined_tf.empty:
            date_lookup = Date[["Version Name", "Transition Start Date"]].drop_duplicates()
            combined_tf = combined_tf.merge(date_lookup, on="Version Name", how="left")
            combined_tf = combined_tf.rename(columns={"Transition Start Date": "Partial Week"})
            combined_tf["Transition Flag"] = 1
            output_tf = combined_tf[output_schema["TransitionFlag"]]

    return {
        "AssortmentFinal": output_af,
        "AssortmentSellOut": output_aso,
        "TransitionFlag": output_tf,
    }

# ------------------------------------------------------------------------------------
# Pytest Setup
# This fixture ensures that `from helpers.plugin_module import main` works in tests.
# ------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_main_module(monkeypatch):
    """
    Creates a mock 'helpers.plugin_module' and injects the 'main' function
    into it, so that tests can import it as required.
    """
    helpers_module = ModuleType("helpers")
    plugin_module = ModuleType("helpers.plugin_module")
    plugin_module.main = main
    
    monkeypatch.setitem(sys.modules, "helpers", helpers_module)
    monkeypatch.setitem(sys.modules, "helpers.plugin_module", plugin_module)

# The import must be moved inside the test functions to ensure the fixture runs first.

# ------------------------------------------------------------------------------------
# Test Helper Functions
# ------------------------------------------------------------------------------------

def df_from_csv_str(csv_string: str) -> pd.DataFrame:
    """Creates a pandas DataFrame from a multi-line CSV string."""
    stripped_csv = csv_string.strip()
    if not stripped_csv:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(stripped_csv), sep=",", skipinitialspace=True)

def assert_df_equal(left: pd.DataFrame, right: pd.DataFrame):
    """
    Asserts that two DataFrames are equal, ignoring index and column order.
    Sorts data to ensure comparison is order-independent.
    """
    if left.empty and right.empty:
        # Both are empty, no need for further checks
        assert left.columns.isin(right.columns).all()
        assert right.columns.isin(left.columns).all()
        return

    # Standardize dtypes for comparison where possible
    for col in left.columns:
        if col in right.columns:
            if left[col].dtype != right[col].dtype:
                try:
                    # Attempt to convert to a common type, like object
                    left[col] = left[col].astype('object').fillna(pd.NA)
                    right[col] = right[col].astype('object').fillna(pd.NA)
                except Exception:
                    pass

    # Sort columns and values
    left_sorted = left.sort_values(by=list(left.columns)).reset_index(drop=True)
    right_sorted = right.sort_values(by=list(right.columns)).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(left_sorted, right_sorted, check_like=True)

# ------------------------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------------------------

def test_golden_record_happy_path():
    from helpers.plugin_module import main
    item_master_csv = """
Transition Item,Planning Item,Item
T1,SKU100_P,SKU100
T1,SKU200_P,SKU200
T1,SKU300_P,SKU300
"""
    assortment_final_csv = """
Version,Version Name,Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location
1,V1,SKU100,Walmart,Walmart_P,Store,Store_P,US,US_P,DD1,PDD1,PnL1,PnL1_P,LOC1
"""
    assortment_sellout_csv = """
Version,Version Name,Item,Planning Item,Account,Planning Account,Channel,Planning Channel,Region,Planning Region,Demand Domain,Planning Demand Domain,PnL,Planning PnL,Location,Planning Location
1,V1,SKU100,SKU100_P,Walmart,Walmart_P,Store,Store_P,US,US_P,DD1,PDD1,PnL1,PnL1_P,LOC1,PLOC1
"""
    date_csv = """
Version Name,Transition Start Date
V1,2024-01-01
"""
    inputs = {
        "ItemMaster": df_from_csv_str(item_master_csv),
        "AssortmentFinal": df_from_csv_str(assortment_final_csv),
        "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv),
        "Date": df_from_csv_str(date_csv),
    }

    results = main(**inputs)

    expected_af = df_from_csv_str("""
Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Location,Assortment Final,Transition Sell In Assortments
V1,SKU200,Walmart_P,Store_P,US_P,SKU200US_P,PnL1_P,LOC1,1,1
V1,SKU300,Walmart_P,Store_P,US_P,SKU300US_P,PnL1_P,LOC1,1,1
""")
    expected_aso = df_from_csv_str("""
Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Mdlz DP Assortment Sell Out,Transition Sell Out Assortments
V1,SKU200_P,Walmart_P,Store_P,US_P,SKU200_PUS_P,PnL1_P,PLOC1,1,1
V1,SKU300_P,Walmart_P,Store_P,US_P,SKU300_PUS_P,PnL1_P,PLOC1,1,1
""")
    expected_tf = df_from_csv_str("""
Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Partial Week,Transition Flag
V1,SKU100_P,Walmart_P,Store_P,US_P,SKU100_PUS_P,PnL1_P,LOC1,2024-01-01,1
V1,SKU200_P,Walmart_P,Store_P,US_P,SKU200_PUS_P,PnL1_P,LOC1,2024-01-01,1
V1,SKU300_P,Walmart_P,Store_P,US_P,SKU300_PUS_P,PnL1_P,LOC1,2024-01-01,1
V1,SKU100_P,Walmart_P,Store_P,US_P,SKU100_PUS_P,PnL1_P,PLOC1,2024-01-01,1
V1,SKU200_P,Walmart_P,Store_P,US_P,SKU200_PUS_P,PnL1_P,PLOC1,2024-01-01,1
V1,SKU300_P,Walmart_P,Store_P,US_P,SKU300_PUS_P,PnL1_P,PLOC1,2024-01-01,1
""")

    assert_df_equal(results["AssortmentFinal"], expected_af)
    assert_df_equal(results["AssortmentSellOut"], expected_aso)
    assert_df_equal(results["TransitionFlag"], expected_tf)

def test_output_schema_and_flag_values():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1\nT1,P2,I2"
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA,PC,PR,PP,L1"
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\nV1,P1,PA,PC,PR,PP,PL1"
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"
    
    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)

    # Schema checks
    expected_af_cols = ["Version Name", "Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Location", "Assortment Final", "Transition Sell In Assortments"]
    expected_aso_cols = ["Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Mdlz DP Assortment Sell Out", "Transition Sell Out Assortments"]
    expected_tf_cols = ["Version Name", "Planning Item", "Planning Account", "Planning Channel", "Planning Region", "Planning Demand Domain", "Planning PnL", "Planning Location", "Partial Week", "Transition Flag"]
    
    assert list(results["AssortmentFinal"].columns) == expected_af_cols
    assert list(results["AssortmentSellOut"].columns) == expected_aso_cols
    assert list(results["TransitionFlag"].columns) == expected_tf_cols

    # Flag value checks
    assert not results["AssortmentFinal"].empty
    assert (results["AssortmentFinal"]["Assortment Final"] == 1).all()
    assert (results["AssortmentFinal"]["Transition Sell In Assortments"] == 1).all()
    
    assert not results["AssortmentSellOut"].empty
    assert (results["AssortmentSellOut"]["Mdlz DP Assortment Sell Out"] == 1).all()
    assert (results["AssortmentSellOut"]["Transition Sell Out Assortments"] == 1).all()
    
    assert not results["TransitionFlag"].empty
    assert (results["TransitionFlag"]["Transition Flag"] == 1).all()

def test_planning_demand_domain_concatenation():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,SKU_XYZ_P,SKU_XYZ"
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,SKU_XYZ,PA,PC,US-WEST,PP,L1"
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\nV1,SKU_XYZ_P,PA,PC,US-WEST,PP,PL1"
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)

    # Note: No new items are generated, so AF/ASO are empty. We check PDD in TransitionFlag.
    tf_row = results["TransitionFlag"].iloc[0]
    # AF PDD uses Item + Planning Region
    # ASO PDD uses Planning Item + Planning Region
    # TF PDD uses Planning Item + Planning Region
    assert tf_row["Planning Demand Domain"] == "SKU_XYZ_PUS-WEST"


def test_date_lookup_for_transition_flag():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1"
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA,PC,PR,PP,L1"
    date_csv = "Version Name,Transition Start Date\nV1,2025-10-31"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": pd.DataFrame(), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)
    
    assert not results["TransitionFlag"].empty
    assert (results["TransitionFlag"]["Partial Week"] == "2025-10-31").all()

def test_empty_assortment_final_input():
    from helpers.plugin_module import main
    # Per logic step 1, seed items come from AssortmentFinal. If it's empty, all outputs must be empty.
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1\nT1,P2,I2"
    assortment_final_csv = ""
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\nV1,P1,PA,PC,PR,PP,PL1"
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)

    assert results["AssortmentFinal"].empty
    assert results["AssortmentSellOut"].empty
    assert results["TransitionFlag"].empty

def test_empty_assortment_sellout_input():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1\nT1,P2,I2"
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA,PC,PR,PP,L1"
    assortment_sellout_csv = ""
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)
    
    expected_af = df_from_csv_str("""
Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Location,Assortment Final,Transition Sell In Assortments
V1,I2,PA,PC,PR,I2PR,PP,L1,1,1
""")
    expected_tf = df_from_csv_str("""
Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Partial Week,Transition Flag
V1,P1,PA,PC,PR,P1PR,PP,L1,2024-01-01,1
V1,P2,PA,PC,PR,P2PR,PP,L1,2024-01-01,1
""")

    assert_df_equal(results["AssortmentFinal"], expected_af)
    assert results["AssortmentSellOut"].empty
    assert_df_equal(results["TransitionFlag"], expected_tf)

def test_all_inputs_empty():
    from helpers.plugin_module import main
    inputs = {
        "ItemMaster": pd.DataFrame(),
        "AssortmentFinal": pd.DataFrame(),
        "AssortmentSellOut": pd.DataFrame(),
        "Date": pd.DataFrame(),
    }
    results = main(**inputs)
    assert results["AssortmentFinal"].empty
    assert results["AssortmentSellOut"].empty
    assert results["TransitionFlag"].empty

def test_no_new_items_in_item_master():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1"
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA,PC,PR,PP,L1"
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\nV1,P1,PA,PC,PR,PP,PL1"
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)

    expected_tf = df_from_csv_str("""
Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Partial Week,Transition Flag
V1,P1,PA,PC,PR,P1PR,PP,L1,2024-01-01,1
V1,P1,PA,PC,PR,P1PR,PP,PL1,2024-01-01,1
""")
    assert results["AssortmentFinal"].empty
    assert results["AssortmentSellOut"].empty
    assert_df_equal(results["TransitionFlag"], expected_tf)
    
def test_all_expanded_rows_already_exist():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1\nT1,P2,I2"
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA,PC,PR,PP,L1\nV1,I2,PA,PC,PR,PP,L1"
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\nV1,P1,PA,PC,PR,PP,PL1\nV1,P2,PA,PC,PR,PP,PL1"
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)

    expected_tf = df_from_csv_str("""
Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Partial Week,Transition Flag
V1,P1,PA,PC,PR,P1PR,PP,L1,2024-01-01,1
V1,P2,PA,PC,PR,P2PR,PP,L1,2024-01-01,1
V1,P1,PA,PC,PR,P1PR,PP,PL1,2024-01-01,1
V1,P2,PA,PC,PR,P2PR,PP,PL1,2024-01-01,1
""")

    assert results["AssortmentFinal"].empty
    assert results["AssortmentSellOut"].empty
    assert_df_equal(results["TransitionFlag"], expected_tf)
    
def test_no_matching_transition_item_in_master():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT99,P99,I99"
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA,PC,PR,PP,L1"
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\nV1,P1,PA,PC,PR,PP,PL1"
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"
    
    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)

    assert results["AssortmentFinal"].empty
    assert results["AssortmentSellOut"].empty
    assert results["TransitionFlag"].empty

def test_missing_date_for_version_name():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1"
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA,PC,PR,PP,L1"
    date_csv = "Version Name,Transition Start Date\nV99,2025-10-31"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": pd.DataFrame(), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)

    assert not results["TransitionFlag"].empty
    assert results["TransitionFlag"]["Partial Week"].isnull().all()
    
def test_null_values_in_grouping_keys():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1"
    # Using an empty string for the null value, pandas will read it as NaN for some dtypes
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA,,PR,PP,L1"
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"

    inputs = {
        "ItemMaster": df_from_csv_str(item_master_csv),
        "AssortmentFinal": df_from_csv_str(assortment_final_csv),
        "AssortmentSellOut": pd.DataFrame(),
        "Date": df_from_csv_str(date_csv),
    }

    results = main(**inputs)
    assert not results["TransitionFlag"].empty
    # The `Planning Channel` should be NaN/None
    tf_result = results["TransitionFlag"]
    assert tf_result["Planning Channel"].isnull().all()
    
def test_multiple_transition_groups():
    from helpers.plugin_module import main
    item_master_csv = """
Transition Item,Planning Item,Item
T1,P1,I1
T1,P2,I2
T2,P10,I10
T2,P11,I11
"""
    assortment_final_csv = """
Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location
V1,I1,PA,PC,PR,PP,L1
V1,I10,PA,PC,PR,PP,L1
"""
    assortment_sellout_csv = ""
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)

    expected_af = df_from_csv_str("""
Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Location,Assortment Final,Transition Sell In Assortments
V1,I2,PA,PC,PR,I2PR,PP,L1,1,1
V1,I11,PA,PC,PR,I11PR,PP,L1,1,1
""")
    assert_df_equal(results["AssortmentFinal"], expected_af)
    assert len(results["TransitionFlag"]) == 4 # I1,I2,I10,I11

def test_multiple_version_names_in_input():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1\nT1,P2,I2"
    assortment_final_csv = """
Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location
V1,I1,PA1,PC1,PR1,PP1,L1
V2,I1,PA2,PC2,PR2,PP2,L2
"""
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01\nV2,2025-02-02"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": pd.DataFrame(), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)
    
    expected_af = df_from_csv_str("""
Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Location,Assortment Final,Transition Sell In Assortments
V1,I2,PA1,PC1,PR1,I2PR1,PP1,L1,1,1
V2,I2,PA2,PC2,PR2,I2PR2,PP2,L2,1,1
""")
    expected_tf = df_from_csv_str("""
Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning Demand Domain,Planning PnL,Planning Location,Partial Week,Transition Flag
V1,P1,PA1,PC1,PR1,P1PR1,PP1,L1,2024-01-01,1
V1,P2,PA1,PC1,PR1,P2PR1,PP1,L1,2024-01-01,1
V2,P1,PA2,PC2,PR2,P1PR2,PP2,L2,2025-02-02,1
V2,P2,PA2,PC2,PR2,P2PR2,PP2,L2,2025-02-02,1
""")
    assert_df_equal(results["AssortmentFinal"], expected_af)
    assert_df_equal(results["TransitionFlag"], expected_tf)

def test_disjoint_assortment_dimensions():
    from helpers.plugin_module import main
    item_master_csv = "Transition Item,Planning Item,Item\nT1,P1,I1\nT1,P2,I2"
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA_final,PC_final,PR_final,PP_final,L_final"
    assortment_sellout_csv = "Version Name,Planning Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Planning Location\nV1,P1,PA_so,PC_so,PR_so,PP_so,PL_so"
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)
    
    assert results["AssortmentFinal"].iloc[0]["Planning Account"] == "PA_final"
    assert results["AssortmentSellOut"].iloc[0]["Planning Account"] == "PA_so"
    
    tf = results["TransitionFlag"]
    assert "PA_final" in tf["Planning Account"].values
    assert "PA_so" in tf["Planning Account"].values

def test_item_master_with_superset_of_items():
    from helpers.plugin_module import main
    item_master_csv = """
Transition Item,Planning Item,Item
T1,P1,I1
T1,P2,I2
T99,P99,I99
"""
    assortment_final_csv = "Version Name,Item,Planning Account,Planning Channel,Planning Region,Planning PnL,Location\nV1,I1,PA,PC,PR,PP,L1"
    assortment_sellout_csv = ""
    date_csv = "Version Name,Transition Start Date\nV1,2024-01-01"

    inputs = {"ItemMaster": df_from_csv_str(item_master_csv), "AssortmentFinal": df_from_csv_str(assortment_final_csv), "AssortmentSellOut": df_from_csv_str(assortment_sellout_csv), "Date": df_from_csv_str(date_csv)}
    results = main(**inputs)

    # Check that only items from T1 group are present
    assert not results["AssortmentFinal"][results["AssortmentFinal"]["Item"] == "I99"].any().any()
    assert not results["TransitionFlag"][results["TransitionFlag"]["Planning Item"] == "P99"].any().any()
    
    # Check that I2 is correctly generated
    assert "I2" in results["AssortmentFinal"]["Item"].values
    assert "P2" in results["TransitionFlag"]["Planning Item"].values