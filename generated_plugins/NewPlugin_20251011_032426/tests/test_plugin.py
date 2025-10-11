from helpers.plugin_module import main
import io
import pandas as pd
import pytest


def assert_df_equal(left, right, key_cols=None):
    """Compares two DataFrames for equality after sorting and resetting index."""
    if key_cols:
        left_sorted = left.sort_values(by=key_cols).reset_index(drop=True)
        right_sorted = right.sort_values(by=key_cols).reset_index(drop=True)
    else:
        left_sorted = left.sort_values(by=left.columns.tolist()).reset_index(drop=True)
        right_sorted = right.sort_values(by=right.columns.tolist()).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(left_sorted, right_sorted, check_dtype=True)


def test_golden_record_single_item():
    """
    Validates that a single item in PlanningItemCustomerGroup with a multi-week date range correctly expands into multiple rows in the Sellin Season output, one for each week.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,Channel A,PnL A,Account A,Domain A,Region A,Location A,Item 1,2023-01-02,2023-01-20
"""
    dim_time_csv = """Date,PartialWeekKey
2023-01-01,202301
2023-01-02,202301
2023-01-08,202302
2023-01-15,202303
2023-01-20,202303
2023-01-22,202304
"""

    planning_item_customer_group = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])

    result_dfs = main(
        PlanningItemCustomerGroup=planning_item_customer_group,
        DimTime=dim_time
    )
    actual_df = result_dfs['Sellin Season']

    expected_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,PartialWeekKey,Sellin Season Week Association
V1,Channel A,PnL A,Account A,Domain A,Region A,Location A,Item 1,202301,1
V1,Channel A,PnL A,Account A,Domain A,Region A,Location A,Item 1,202302,1
V1,Channel A,PnL A,Account A,Domain A,Region A,Location A,Item 1,202303,1
"""
    expected_df = pd.read_csv(io.StringIO(expected_csv))
    
    output_cols = [
        "Version Name", "Planning Channel", "Planning PnL", "Planning Account",
        "Planning Demand Domain", "Planning Region", "Planning Location", "Planning Item",
        "PartialWeekKey", "Sellin Season Week Association"
    ]
    
    assert_df_equal(
        actual_df,
        expected_df,
        key_cols=output_cols
    )


def test_golden_record_multiple_items_same_dates():
    """
    Tests with multiple, distinct planning items that share the exact same Intro and Disco dates. The output should correctly expand all items for the common week range.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,Channel A,PnL A,Account A,Domain A,Region A,Location A,Item X,2023-02-06,2023-02-19
V1,Channel B,PnL B,Account B,Domain B,Region B,Location B,Item Y,2023-02-06,2023-02-19
"""
    dim_time_csv = """Date,PartialWeekKey
2023-02-05,202306
2023-02-06,202306
2023-02-12,202307
2023-02-19,202308
2023-02-20,202308
"""

    planning_item_customer_group = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])

    result_dfs = main(
        PlanningItemCustomerGroup=planning_item_customer_group,
        DimTime=dim_time
    )
    actual_df = result_dfs['Sellin Season']

    expected_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,PartialWeekKey,Sellin Season Week Association
V1,Channel A,PnL A,Account A,Domain A,Region A,Location A,Item X,202306,1
V1,Channel A,PnL A,Account A,Domain A,Region A,Location A,Item X,202307,1
V1,Channel A,PnL A,Account A,Domain A,Region A,Location A,Item X,202308,1
V1,Channel B,PnL B,Account B,Domain B,Region B,Location B,Item Y,202306,1
V1,Channel B,PnL B,Account B,Domain B,Region B,Location B,Item Y,202307,1
V1,Channel B,PnL B,Account B,Domain B,Region B,Location B,Item Y,202308,1
"""
    expected_df = pd.read_csv(io.StringIO(expected_csv))

    output_cols = [
        "Version Name", "Planning Channel", "Planning PnL", "Planning Account",
        "Planning Demand Domain", "Planning Region", "Planning Location", "Planning Item",
        "PartialWeekKey", "Sellin Season Week Association"
    ]

    assert_df_equal(
        actual_df,
        expected_df,
        key_cols=output_cols
    )


def test_golden_record_multiple_items_overlapping_dates():
    """
    Validates the logic with multiple items having different but overlapping date ranges, ensuring each item's weekly expansion is independent and correct.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,Channel C,PnL C,Account C,Domain C,Region C,Location C,Item P,2023-03-01,2023-03-15
V1,Channel D,PnL D,Account D,Domain D,Region D,Location D,Item Q,2023-03-10,2023-03-25
"""
    dim_time_csv = """Date,PartialWeekKey
2023-03-01,202309
2023-03-08,202310
2023-03-10,202310
2023-03-15,202311
2023-03-22,202312
2023-03-25,202312
"""

    planning_item_customer_group = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])

    result_dfs = main(
        PlanningItemCustomerGroup=planning_item_customer_group,
        DimTime=dim_time
    )
    actual_df = result_dfs['Sellin Season']

    expected_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,PartialWeekKey,Sellin Season Week Association
V1,Channel C,PnL C,Account C,Domain C,Region C,Location C,Item P,202309,1
V1,Channel C,PnL C,Account C,Domain C,Region C,Location C,Item P,202310,1
V1,Channel C,PnL C,Account C,Domain C,Region C,Location C,Item P,202311,1
V1,Channel D,PnL D,Account D,Domain D,Region D,Location D,Item Q,202310,1
V1,Channel D,PnL D,Account D,Domain D,Region D,Location D,Item Q,202311,1
V1,Channel D,PnL D,Account D,Domain D,Region D,Location D,Item Q,202312,1
"""
    expected_df = pd.read_csv(io.StringIO(expected_csv))

    output_cols = [
        "Version Name", "Planning Channel", "Planning PnL", "Planning Account",
        "Planning Demand Domain", "Planning Region", "Planning Location", "Planning Item",
        "PartialWeekKey", "Sellin Season Week Association"
    ]

    assert_df_equal(
        actual_df,
        expected_df,
        key_cols=output_cols
    )


def test_golden_record_date_range_spans_multiple_week_keys():
    """
    Ensures that a date range correctly identifies and includes all distinct PartialWeekKeys that fall within it.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,Channel E,PnL E,Account E,Domain E,Region E,Location E,Item Z,2023-05-01,2023-05-14
"""
    dim_time_csv = """Date,PartialWeekKey
2023-05-01,202318
2023-05-02,202318
2023-05-06,202318
2023-05-07,202319
2023-05-08,202319
2023-05-13,202319
2023-05-14,202320
2023-05-15,202320
"""

    planning_item_customer_group = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])

    result_dfs = main(
        PlanningItemCustomerGroup=planning_item_customer_group,
        DimTime=dim_time
    )
    actual_df = result_dfs['Sellin Season']

    expected_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,PartialWeekKey,Sellin Season Week Association
V1,Channel E,PnL E,Account E,Domain E,Region E,Location E,Item Z,202318,1
V1,Channel E,PnL E,Account E,Domain E,Region E,Location E,Item Z,202319,1
V1,Channel E,PnL E,Account E,Domain E,Region E,Location E,Item Z,202320,1
"""
    expected_df = pd.read_csv(io.StringIO(expected_csv))

    output_cols = [
        "Version Name", "Planning Channel", "Planning PnL", "Planning Account",
        "Planning Demand Domain", "Planning Region", "Planning Location", "Planning Item",
        "PartialWeekKey", "Sellin Season Week Association"
    ]

    assert_df_equal(
        actual_df,
        expected_df,
        key_cols=output_cols
    )

def test_output_schema_and_constant_value():
    """
    Verifies that the final output table contains all specified columns in the correct format
    and that the 'Sellin Season Week Association' column is always populated with the value 1.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,Channel1,PnL1,Account1,DD1,Region1,Location1,ItemA,2023-01-10,2023-01-25
V2,Channel2,PnL2,Account2,DD2,Region2,Location2,ItemB,2023-02-01,2023-02-05
"""
    dim_time_csv = """Date,PartialWeekKey
2023-01-15,202303
2023-01-22,202304
2023-02-19,202308
"""
    input_data = {
        "PlanningItemCustomerGroup": pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date']),
        "DimTime": pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    }

    result_dfs = main(**input_data)
    output_df = result_dfs["Sellin Season"]

    expected_schema = [
        "Version Name",
        "Planning Channel",
        "Planning PnL",
        "Planning Account",
        "Planning Demand Domain",
        "Planning Region",
        "Planning Location",
        "Planning Item",
        "PartialWeekKey",
        "Sellin Season Week Association"
    ]

    assert sorted(list(output_df.columns)) == sorted(expected_schema)
    assert not output_df.empty
    assert (output_df["Sellin Season Week Association"] == 1).all()


def test_golden_record_duplicate_item_entries():
    """
    Tests the scenario where the input PlanningItemCustomerGroup contains fully duplicate rows.
    The process should treat them as distinct and generate expanded weekly rows for both.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,CH1,PNL1,ACC1,DD1,REG1,LOC1,ItemA,2023-05-01,2023-05-20
V1,CH1,PNL1,ACC1,DD1,REG1,LOC1,ItemA,2023-05-01,2023-05-20
"""
    dim_time_csv = """Date,PartialWeekKey
2023-05-07,202319
2023-05-14,202320
"""
    input_data = {
        "PlanningItemCustomerGroup": pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date']),
        "DimTime": pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    }

    result_dfs = main(**input_data)
    output_df = result_dfs["Sellin Season"]

    # Each of the 2 duplicate input rows should be expanded by the 2 valid weeks. 2 * 2 = 4 rows.
    assert len(output_df) == 4
    # Verify the correct number of unique items and weeks are present
    assert len(output_df.drop_duplicates(subset=["Planning Item", "PartialWeekKey"])) == 2
    # Verify both weeks are present in the output
    assert set(output_df["PartialWeekKey"]) == {202319, 202320}


def test_empty_planning_item_customer_group_input():
    """
    Tests the behavior when the PlanningItemCustomerGroup input table is empty.
    The output Sellin Season table should also be empty.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
"""
    dim_time_csv = """Date,PartialWeekKey
2023-01-01,202301
"""
    input_data = {
        "PlanningItemCustomerGroup": pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date']),
        "DimTime": pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    }

    result_dfs = main(**input_data)
    output_df = result_dfs["Sellin Season"]

    expected_schema = [
        "Version Name", "Planning Channel", "Planning PnL", "Planning Account",
        "Planning Demand Domain", "Planning Region", "Planning Location",
        "Planning Item", "PartialWeekKey", "Sellin Season Week Association"
    ]

    assert output_df.empty
    assert sorted(list(output_df.columns)) == sorted(expected_schema)


def test_empty_dim_time_input():
    """
    Tests the behavior when the DimTime input table is empty. The output should be empty,
    as no weeks can be generated, even if PlanningItemCustomerGroup has data.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,CH1,PNL1,ACC1,DD1,REG1,LOC1,ItemA,2023-03-01,2023-03-15
"""
    dim_time_csv = """Date,PartialWeekKey
"""
    input_data = {
        "PlanningItemCustomerGroup": pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date']),
        "DimTime": pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    }

    result_dfs = main(**input_data)
    output_df = result_dfs["Sellin Season"]

    expected_schema = [
        "Version Name", "Planning Channel", "Planning PnL", "Planning Account",
        "Planning Demand Domain", "Planning Region", "Planning Location",
        "Planning Item", "PartialWeekKey", "Sellin Season Week Association"
    ]

    assert output_df.empty
    assert sorted(list(output_df.columns)) == sorted(expected_schema)

def test_intro_date_equals_disco_date():
    """
    Tests an item with an Intro Date and Disco Date that are the same.
    The output should contain exactly one row for that item, corresponding to the week of that single day.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,ChannelA,PnL1,AccountX,DD1,Global,USA,ITEM-101,2023-01-03,2023-01-03
"""
    dim_time_csv = """Date,PartialWeekKey
2023-01-01,202301
2023-01-02,202301
2023-01-03,202301
2023-01-04,202301
2023-01-08,202302
2023-01-09,202302
"""
    expected_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,PartialWeekKey,Sellin Season Week Association
V1,ChannelA,PnL1,AccountX,DD1,Global,USA,ITEM-101,202301,1
"""

    planning_item_customer_group = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    expected_output = pd.read_csv(io.StringIO(expected_csv))
    
    input_data = {
        'PlanningItemCustomerGroup': planning_item_customer_group,
        'DimTime': dim_time
    }
    
    result_data = main(**input_data)
    actual_output = result_data['Sellin Season']

    pd.testing.assert_frame_equal(actual_output.reset_index(drop=True), expected_output.reset_index(drop=True), check_like=True)


def test_intro_date_after_disco_date():
    """
    Tests an invalid record where the Intro Date is after the Disco Date.
    This record should produce zero rows in the output.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,ChannelB,PnL2,AccountY,DD2,EMEA,Germany,ITEM-202,2023-05-10,2023-05-01
"""
    dim_time_csv = """Date,PartialWeekKey
2023-05-01,202318
2023-05-02,202318
2023-05-10,202319
"""

    planning_item_customer_group = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])

    output_schema = {
        "Version Name": "object",
        "Planning Channel": "object",
        "Planning PnL": "object",
        "Planning Account": "object",
        "Planning Demand Domain": "object",
        "Planning Region": "object",
        "Planning Location": "object",
        "Planning Item": "object",
        "PartialWeekKey": "int64",
        "Sellin Season Week Association": "int64"
    }
    expected_output = pd.DataFrame(columns=output_schema.keys()).astype(output_schema)

    input_data = {
        'PlanningItemCustomerGroup': planning_item_customer_group,
        'DimTime': dim_time
    }

    result_data = main(**input_data)
    actual_output = result_data['Sellin Season']
    
    pd.testing.assert_frame_equal(actual_output, expected_output, check_like=True)


def test_null_intro_or_disco_date():
    """
    Tests how the logic handles rows in PlanningItemCustomerGroup with a NULL Intro Date or Disco Date.
    These rows should be gracefully ignored and not appear in the output.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,CH1,PNL1,ACC1,DD1,REG1,LOC1,ITEM_VALID,2023-07-10,2023-07-11
V1,CH1,PNL1,ACC1,DD1,REG1,LOC1,ITEM_NULL_INTRO,,2023-07-11
V1,CH1,PNL1,ACC1,DD1,REG1,LOC1,ITEM_NULL_DISCO,2023-07-10,
V1,CH1,PNL1,ACC1,DD1,REG1,LOC1,ITEM_NULL_BOTH,,
"""
    dim_time_csv = """Date,PartialWeekKey
2023-07-10,202328
2023-07-11,202328
"""
    expected_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,PartialWeekKey,Sellin Season Week Association
V1,CH1,PNL1,ACC1,DD1,REG1,LOC1,ITEM_VALID,202328,1
"""
    
    planning_item_customer_group = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    expected_output = pd.read_csv(io.StringIO(expected_csv))
    
    input_data = {
        'PlanningItemCustomerGroup': planning_item_customer_group,
        'DimTime': dim_time
    }

    result_data = main(**input_data)
    actual_output = result_data['Sellin Season']
    
    pd.testing.assert_frame_equal(actual_output.reset_index(drop=True), expected_output.reset_index(drop=True), check_like=True)


def test_date_range_outside_dim_time_scope():
    """
    Tests an item whose date range is entirely before or after the dates available in the DimTime table.
    This item should not generate any rows in the output.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,ChannelC,PnL3,AccountZ,DD3,APAC,Japan,ITEM-303,2022-12-20,2022-12-25
"""
    dim_time_csv = """Date,PartialWeekKey
2023-01-01,202301
2023-01-02,202301
2023-01-03,202301
"""

    planning_item_customer_group = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    
    output_schema = {
        "Version Name": "object",
        "Planning Channel": "object",
        "Planning PnL": "object",
        "Planning Account": "object",
        "Planning Demand Domain": "object",
        "Planning Region": "object",
        "Planning Location": "object",
        "Planning Item": "object",
        "PartialWeekKey": "int64",
        "Sellin Season Week Association": "int64"
    }
    expected_output = pd.DataFrame(columns=output_schema.keys()).astype(output_schema)
    
    input_data = {
        'PlanningItemCustomerGroup': planning_item_customer_group,
        'DimTime': dim_time
    }
    
    result_data = main(**input_data)
    actual_output = result_data['Sellin Season']
    
    pd.testing.assert_frame_equal(actual_output, expected_output, check_like=True)

def test_inclusive_date_boundaries():
    """
    Verifies that the date range expansion is inclusive, meaning if an Intro or Disco Date is on a given day, the PartialWeekKey for that day is included in the result.
    """
    planning_item_cg_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,Retail,PnL1,Account1,Domain1,Region1,Location1,ItemA,2023-01-15,2023-01-17
"""
    dim_time_csv = """Date,PartialWeekKey
2023-01-14,202302
2023-01-15,202303
2023-01-16,202303
2023-01-17,202303
2023-01-18,202303
"""
    input_data = {
        "PlanningItemCustomerGroup": pd.read_csv(io.StringIO(planning_item_cg_csv), parse_dates=['Intro Date', 'Disco Date']),
        "DimTime": pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    }

    result_dfs = main(**input_data)
    output_df = result_dfs["Sellin Season"]

    assert len(output_df) == 1
    assert output_df["PartialWeekKey"].min() == 202303
    assert output_df["PartialWeekKey"].max() == 202303
    assert (output_df["Sellin Season Week Association"] == 1).all()
    assert set(output_df["PartialWeekKey"]) == {202303}


def test_date_range_partially_overlaps_dim_time():
    """
    Tests an item whose date range only partially overlaps with the dates in DimTime. The output should only contain rows for the weeks within the overlapping period.
    """
    planning_item_cg_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,Online,PnL2,Account2,Domain2,Region2,Location2,ItemB,2023-05-10,2023-05-20
"""
    dim_time_csv = """Date,PartialWeekKey
2023-05-01,202318
2023-05-15,202320
2023-05-16,202320
2023-05-17,202320
2023-05-18,202320
2023-05-19,202320
2023-05-20,202321
2023-05-21,202321
2023-05-25,202322
"""
    input_data = {
        "PlanningItemCustomerGroup": pd.read_csv(io.StringIO(planning_item_cg_csv), parse_dates=['Intro Date', 'Disco Date']),
        "DimTime": pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    }

    result_dfs = main(**input_data)
    output_df = result_dfs["Sellin Season"]

    assert len(output_df) == 2
    assert output_df["PartialWeekKey"].min() == 202320
    assert output_df["PartialWeekKey"].max() == 202321
    assert output_df["Planning Item"].iloc[0] == "ItemB"
    assert (output_df["Sellin Season Week Association"] == 1).all()


def test_very_long_date_range():
    """
    Tests the system's performance and correctness with an item that has a date range spanning several years, which should generate a large number of weekly rows.
    Note: A one-month period is used as a proxy for a multi-year range to keep test data manageable.
    """
    planning_item_cg_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,Retail,PnL_Long,Acc_Long,Dom_Long,Reg_Long,Loc_Long,Item_Long,2023-03-01,2023-03-31
"""
    dim_time_days = [f"2023-03-{day:02d},{2309 + (day-1)//7}" for day in range(1, 32)]
    dim_time_csv = "Date,PartialWeekKey\n" + "\n".join(dim_time_days)

    input_data = {
        "PlanningItemCustomerGroup": pd.read_csv(io.StringIO(planning_item_cg_csv), parse_dates=['Intro Date', 'Disco Date']),
        "DimTime": pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    }

    result_dfs = main(**input_data)
    output_df = result_dfs["Sellin Season"]

    assert len(output_df) == 5
    assert output_df["PartialWeekKey"].min() == 2309
    assert output_df["PartialWeekKey"].max() == 2313
    assert output_df["Planning Item"].iloc[0] == "Item_Long"
    assert (output_df["Sellin Season Week Association"] == 1).all()


def test_many_unique_date_ranges():
    """
    Tests the efficiency of identifying unique date ranges when the input contains a large number of different Intro/Disco Date pairs.
    """
    planning_item_cg_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,Retail,PnL1,Account1,Domain1,Region1,Location1,Item1,2023-04-01,2023-04-03
V1,Retail,PnL1,Account1,Domain1,Region1,Location1,Item2,2023-04-01,2023-04-03
V1,Online,PnL2,Account2,Domain2,Region2,Location2,Item3,2023-04-10,2023-04-12
V1,Online,PnL2,Account2,Domain2,Region2,Location2,Item4,2023-04-10,2023-04-12
"""
    dim_time_csv = """Date,PartialWeekKey
2023-04-01,202314
2023-04-02,202314
2023-04-03,202315
2023-04-08,202315
2023-04-09,202315
2023-04-10,202316
2023-04-11,202316
2023-04-12,202316
"""
    input_data = {
        "PlanningItemCustomerGroup": pd.read_csv(io.StringIO(planning_item_cg_csv), parse_dates=['Intro Date', 'Disco Date']),
        "DimTime": pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])
    }

    result_dfs = main(**input_data)
    output_df = result_dfs["Sellin Season"]

    # (2 items * 2 unique weeks for first range) + (2 items * 1 unique week for second range) = 4 + 2 = 6
    assert len(output_df) == 6

    # Check first unique range expansion
    df_range1 = output_df[output_df["Planning Item"].isin(["Item1", "Item2"])]
    assert len(df_range1) == 4 # 2 items * 2 weeks
    assert set(df_range1["Planning Item"]) == {"Item1", "Item2"}
    assert set(df_range1["PartialWeekKey"]) == {202314, 202315}

    # Check second unique range expansion
    df_range2 = output_df[output_df["Planning Item"].isin(["Item3", "Item4"])]
    assert len(df_range2) == 2 # 2 items * 1 week
    assert set(df_range2["Planning Item"]) == {"Item3", "Item4"}
    assert set(df_range2["PartialWeekKey"]) == {202316}

    assert (output_df["Sellin Season Week Association"] == 1).all()

def test_input_with_extra_unspecified_columns():
    """
    Ensures that if the input tables have additional columns not mentioned in the logic,
    they are ignored and do not affect the output or cause errors.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date,Extra Planning Col
V1,C1,P1,A1,DD1,R1,L1,ITEM001,2023-05-01,2023-05-03,ExtraVal1
"""
    dim_time_csv = """Date,PartialWeekKey,Extra Time Col
2023-05-01,202318,Monday
2023-05-02,202318,Tuesday
2023-05-03,202318,Wednesday
2023-05-04,202318,Thursday
"""
    planning_item_customer_group_df = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time_df = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])

    result = main(
        PlanningItemCustomerGroup=planning_item_customer_group_df,
        DimTime=dim_time_df
    )
    actual_df = result["Sellin Season"]

    expected_data = [
        {'Version Name': 'V1', 'Planning Channel': 'C1', 'Planning PnL': 'P1', 'Planning Account': 'A1', 'Planning Demand Domain': 'DD1', 'Planning Region': 'R1', 'Planning Location': 'L1', 'Planning Item': 'ITEM001', 'PartialWeekKey': 202318, 'Sellin Season Week Association': 1},
    ]
    expected_df = pd.DataFrame(expected_data)

    # Ensure dtypes match for comparison
    expected_df['Sellin Season Week Association'] = expected_df['Sellin Season Week Association'].astype('int64')
    expected_df['PartialWeekKey'] = expected_df['PartialWeekKey'].astype('int64')

    pd.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True), check_like=True)


def test_non_contiguous_dim_time():
    """
    Tests the logic against a DimTime table that has gaps in its DayKey sequence.
    The expansion should correctly skip the missing days and only use existing PartialWeekKeys.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V1,C1,P1,A1,DD1,R1,L1,ITEM001,2023-10-28,2023-11-02
"""
    # DimTime is missing 2023-10-29, 2023-10-31, 2023-11-01
    dim_time_csv = """Date,PartialWeekKey
2023-10-28,202343
2023-10-30,202344
2023-11-02,202344
2023-11-03,202344
"""
    planning_item_customer_group_df = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time_df = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])

    result = main(
        PlanningItemCustomerGroup=planning_item_customer_group_df,
        DimTime=dim_time_df
    )
    actual_df = result["Sellin Season"]

    expected_data = [
        {'Version Name': 'V1', 'Planning Channel': 'C1', 'Planning PnL': 'P1', 'Planning Account': 'A1', 'Planning Demand Domain': 'DD1', 'Planning Region': 'R1', 'Planning Location': 'L1', 'Planning Item': 'ITEM001', 'PartialWeekKey': 202343, 'Sellin Season Week Association': 1},
        {'Version Name': 'V1', 'Planning Channel': 'C1', 'Planning PnL': 'P1', 'Planning Account': 'A1', 'Planning Demand Domain': 'DD1', 'Planning Region': 'R1', 'Planning Location': 'L1', 'Planning Item': 'ITEM001', 'PartialWeekKey': 202344, 'Sellin Season Week Association': 1},
    ]
    expected_df = pd.DataFrame(expected_data)

    expected_df['Sellin Season Week Association'] = expected_df['Sellin Season Week Association'].astype('int64')
    expected_df['PartialWeekKey'] = expected_df['PartialWeekKey'].astype('int64')


    assert_df_equal(actual_df, expected_df, key_cols=['PartialWeekKey'])


def test_date_range_spanning_leap_year():
    """
    Validates that a date range that includes a leap day (February 29th)
    is handled correctly and week generation is not affected.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
V_LEAP,C_LEAP,P_LEAP,A_LEAP,DD_LEAP,R_LEAP,L_LEAP,ITEM_LEAP,2024-02-28,2024-03-02
"""
    dim_time_csv = """Date,PartialWeekKey
2024-02-27,202409
2024-02-28,202409
2024-02-29,202409
2024-03-01,202409
2024-03-02,202409
2024-03-03,202409
"""
    planning_item_customer_group_df = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time_df = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])

    result = main(
        PlanningItemCustomerGroup=planning_item_customer_group_df,
        DimTime=dim_time_df
    )
    actual_df = result["Sellin Season"]

    expected_data = [
        {'Version Name': 'V_LEAP', 'Planning Channel': 'C_LEAP', 'Planning PnL': 'P_LEAP', 'Planning Account': 'A_LEAP', 'Planning Demand Domain': 'DD_LEAP', 'Planning Region': 'R_LEAP', 'Planning Location': 'L_LEAP', 'Planning Item': 'ITEM_LEAP', 'PartialWeekKey': 202409, 'Sellin Season Week Association': 1},
    ]
    expected_df = pd.DataFrame(expected_data)

    expected_df['Sellin Season Week Association'] = expected_df['Sellin Season Week Association'].astype('int64')
    expected_df['PartialWeekKey'] = expected_df['PartialWeekKey'].astype('int64')


    pd.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True), check_like=True)


def test_all_planning_fields_null():
    """
    Tests an item with valid dates but NULL values for all other planning dimension columns
    (Version, Channel, PnL, etc.). The output should still generate weekly rows with
    NULLs for those dimension columns.
    """
    planning_item_customer_group_csv = """Version Name,Planning Channel,Planning PnL,Planning Account,Planning Demand Domain,Planning Region,Planning Location,Planning Item,Intro Date,Disco Date
,,,,,,,,2023-07-10,2023-07-12
"""
    dim_time_csv = """Date,PartialWeekKey
2023-07-10,202328
2023-07-11,202328
2023-07-12,202328
"""
    planning_item_customer_group_df = pd.read_csv(io.StringIO(planning_item_customer_group_csv), parse_dates=['Intro Date', 'Disco Date'])
    dim_time_df = pd.read_csv(io.StringIO(dim_time_csv), parse_dates=['Date'])

    result = main(
        PlanningItemCustomerGroup=planning_item_customer_group_df,
        DimTime=dim_time_df
    )
    actual_df = result["Sellin Season"]

    expected_data = [
        {'Version Name': None, 'Planning Channel': None, 'Planning PnL': None, 'Planning Account': None, 'Planning Demand Domain': None, 'Planning Region': None, 'Planning Location': None, 'Planning Item': None, 'PartialWeekKey': 202328, 'Sellin Season Week Association': 1},
    ]
    
    # Use the expected output schema order
    output_columns = [
        "Version Name", "Planning Channel", "Planning PnL", "Planning Account",
        "Planning Demand Domain", "Planning Region", "Planning Location", "Planning Item",
        "PartialWeekKey", "Sellin Season Week Association"
    ]
    expected_df = pd.DataFrame(expected_data, columns=output_columns)
    
    # Set dtypes correctly for comparison
    expected_df['Sellin Season Week Association'] = expected_df['Sellin Season Week Association'].astype('int64')
    expected_df['PartialWeekKey'] = expected_df['PartialWeekKey'].astype('int64')

    for col in expected_df.columns:
        if col not in ['PartialWeekKey', 'Sellin Season Week Association']:
            expected_df[col] = expected_df[col].astype(object).where(pd.notna(expected_df[col]), None)

    pd.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True), check_like=True)