# file: plugin_module.py
import logging

import pandas as pd

# Initialize logger
logger = logging.getLogger("o9_logger")


def main(PlanningItemCustomerGroup: pd.DataFrame, DimTime: pd.DataFrame) -> dict:
    """
    Core logic to generate the Sellin Season week association.

    Args:
        PlanningItemCustomerGroup (pd.DataFrame): DataFrame containing item-customer group
                                                  information with Intro and Disco dates.
        DimTime (pd.DataFrame): DataFrame containing time dimension data, with Date and PartialWeekKey.

    Returns:
        dict: A dictionary with a single key 'Sellin Season' containing a DataFrame
              with item-customer groups expanded for each active week.
    """
    # Define constants for column names
    _INTRO_DATE = "Intro Date"
    _DISCO_DATE = "Disco Date"
    _PW_KEY = "PartialWeekKey"
    _DATE = "Date"
    _SELLIN_ASSOC = "Sellin Season Week Association"
    _VERSION_NAME = "Version Name"
    _PLANNING_CHANNEL = "Planning Channel"
    _PLANNING_PNL = "Planning PnL"
    _PLANNING_ACCOUNT = "Planning Account"
    _PLANNING_DEMAND_DOMAIN = "Planning Demand Domain"
    _PLANNING_REGION = "Planning Region"
    _PLANNING_LOCATION = "Planning Location"
    _PLANNING_ITEM = "Planning Item"

    # Define the output schema with dtypes for robust empty DataFrame creation and final casting
    output_schema = {
        _VERSION_NAME: "object",
        _PLANNING_CHANNEL: "object",
        _PLANNING_PNL: "object",
        _PLANNING_ACCOUNT: "object",
        _PLANNING_DEMAND_DOMAIN: "object",
        _PLANNING_REGION: "object",
        _PLANNING_LOCATION: "object",
        _PLANNING_ITEM: "object",
        _PW_KEY: "int64",
        _SELLIN_ASSOC: "int64"
    }
    output_cols = list(output_schema.keys())

    # Helper function to create an empty DF with correct schema
    def create_empty_df():
        return pd.DataFrame(columns=output_cols).astype(output_schema)

    # 1. Handle empty inputs gracefully
    if PlanningItemCustomerGroup.empty or DimTime.empty:
        logger.warning("One or both input DataFrames are empty. Returning an empty DataFrame.")
        return {"Sellin Season": create_empty_df()}
    
    if _DATE not in DimTime.columns or _PW_KEY not in DimTime.columns:
        logger.error(f"DimTime is missing required columns: '{_DATE}' and/or '{_PW_KEY}'.")
        return {"Sellin Season": create_empty_df()}

    # Create copies to avoid modifying original DataFrames
    planning_df = PlanningItemCustomerGroup.copy()
    time_df = DimTime.copy()

    # 2. Prepare data: drop null dates and convert to datetime objects for comparison
    planning_df.dropna(subset=[_INTRO_DATE, _DISCO_DATE], inplace=True)
    if planning_df.empty:
        logger.warning("No rows with valid Intro and Disco dates found. Returning an empty DataFrame.")
        return {"Sellin Season": create_empty_df()}

    try:
        planning_df[_INTRO_DATE] = pd.to_datetime(planning_df[_INTRO_DATE])
        planning_df[_DISCO_DATE] = pd.to_datetime(planning_df[_DISCO_DATE])
        time_df[_DATE] = pd.to_datetime(time_df[_DATE])
    except Exception as e:
        logger.error(f"Error converting date columns to datetime: {e}")
        return {"Sellin Season": create_empty_df()}
        
    # Filter out invalid date ranges where intro > disco
    planning_df = planning_df[planning_df[_INTRO_DATE] <= planning_df[_DISCO_DATE]].copy()
    if planning_df.empty:
        logger.warning("No valid date ranges (Intro <= Disco) found. Returning an empty DataFrame.")
        return {"Sellin Season": create_empty_df()}

    # 3. Step 1 from pseudocode: Create UniqueDateRanges
    logger.info("Creating unique date ranges.")
    unique_date_ranges = planning_df[[_INTRO_DATE, _DISCO_DATE]].drop_duplicates().reset_index(drop=True)

    # Prepare a unique mapping of dates to week keys from DimTime
    time_map_df = time_df[[_DATE, _PW_KEY]].drop_duplicates()

    # 4. Step 2 from pseudocode: Create ExpandedWeeks
    logger.info("Expanding date ranges to weeks using a cross merge.")
    if unique_date_ranges.empty or time_map_df.empty:
        logger.warning("No unique date ranges or time data to process. Returning an empty DataFrame.")
        return {"Sellin Season": create_empty_df()}

    expanded_weeks = unique_date_ranges.merge(time_map_df, how='cross')

    # Filter to keep only the days within each intro/disco range
    if not expanded_weeks.empty:
        expanded_weeks = expanded_weeks[
            (expanded_weeks[_DATE] >= expanded_weeks[_INTRO_DATE]) &
            (expanded_weeks[_DATE] <= expanded_weeks[_DISCO_DATE])
        ]

    # After filtering, we only need the Intro/Disco/PartialWeekKey combination.
    # Drop duplicates to ensure each range maps to a unique set of week keys.
    if not expanded_weeks.empty:
        expanded_weeks = expanded_weeks[[_INTRO_DATE, _DISCO_DATE, _PW_KEY]].drop_duplicates()
    else:
        logger.warning("Date range expansion resulted in no matching weeks. Returning an empty DataFrame.")
        return {"Sellin Season": create_empty_df()}

    # 5. Step 3 from pseudocode: Create ItemWeeks by joining back
    logger.info("Joining expanded weeks back to the planning data.")
    item_weeks = pd.merge(
        planning_df,
        expanded_weeks,
        on=[_INTRO_DATE, _DISCO_DATE],
        how='inner'
    )

    if item_weeks.empty:
        logger.warning("No matching weeks found for any items. Returning an empty DataFrame.")
        return {"Sellin Season": create_empty_df()}

    # 6. Step 4 & 5 from pseudocode: Select columns
    logger.info("Formatting the final output table.")
    final_dims = [col for col in output_cols if col != _SELLIN_ASSOC]
    
    missing_cols = [col for col in final_dims if col not in item_weeks.columns]
    if missing_cols:
        logger.error(f"The following required columns are missing from the merged data: {missing_cols}")
        raise KeyError(f"Required columns not found in data: {missing_cols}")
        
    sellin_season = item_weeks[final_dims]

    # 7. Step 6 from pseudocode: Add the association measure
    sellin_season = sellin_season.assign(**{_SELLIN_ASSOC: 1})
    
    # Reorder to final schema and enforce dtypes
    sellin_season = sellin_season[output_cols].astype(output_schema)
    
    # Convert NaN in object columns to None for cleaner output, consistent with tests
    for col in sellin_season.select_dtypes(include=['object']).columns:
        sellin_season[col] = sellin_season[col].where(pd.notna(sellin_season[col]), None)

    # No need for a final drop_duplicates(), as it would incorrectly remove intentional duplicates
    # from the source PlanningItemCustomerGroup data.
    sellin_season.reset_index(drop=True, inplace=True)

    logger.info(f"Generated {len(sellin_season)} rows for Sellin Season association.")
    
    return {"Sellin Season": sellin_season}