import time

import numpy as np
import pandas as pd
from o9Reference.API.member_creation import create_members
from o9Reference.API.o9api import O9API
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import remove_first_brackets

from helpers.o9Constants import o9Constants
from helpers.o9helpers.o9exception import InputEmptyException
from helpers.o9helpers.o9logger import O9Logger

logger = O9Logger()

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


class Constants:
    """Configuration for all column names and constants"""

    # --- Dimensions  ---
    VERSION_COL = o9Constants.VERSION_NAME
    DATA_OBJECT = o9Constants.DATA_OBJECT
    DM_RULE = o9Constants.DM_RULE
    SEQUENCE_COL = o9Constants.SEQUENCE
    PERSONNEL_EMAIL = o9Constants.PERSONNEL_EMAIL

    # --- Input UI Columns ---
    FROM_ITEM_UI = o9Constants.DP_FROM_ITEM_SCOPE_UI_INPUT
    FROM_LOCATION_UI = o9Constants.DP_FROM_LOCATION_SCOPE_UI_INPUT
    FROM_CHANNEL_UI = o9Constants.DP_FROM_CHANNEL_SCOPE_UI_INPUT
    FROM_ACCOUNT_UI = o9Constants.DP_FROM_ACCOUNT_SCOPE_UI_INPUT
    FROM_PNL_UI = o9Constants.DP_FROM_PNL_SCOPE_UI_INPUT
    FROM_DEMAND_DOMAIN_UI = o9Constants.DP_FROM_DEMAND_DOMAIN_SCOPE_UI_INPUT
    FROM_REGION_UI = o9Constants.DP_FROM_REGION_SCOPE_UI_INPUT

    TO_ITEM_UI = o9Constants.DP_TO_ITEM_SCOPE_UI_INPUT
    TO_LOCATION_UI = o9Constants.DP_TO_LOCATION_SCOPE_UI_INPUT
    TO_CHANNEL_UI = o9Constants.DP_TO_CHANNEL_SCOPE_UI_INPUT
    TO_ACCOUNT_UI = o9Constants.DP_TO_ACCOUNT_SCOPE_UI_INPUT
    TO_PNL_UI = o9Constants.DP_TO_PNL_SCOPE_UI_INPUT
    TO_DEMAND_DOMAIN_UI = o9Constants.DP_TO_DEMAND_DOMAIN_SCOPE_UI_INPUT
    TO_REGION_UI = o9Constants.DP_TO_REGION_SCOPE_UI_INPUT

    REALIGNMENT_DATA_OBJECT_UI = o9Constants.DP_REALIGNMENT_DATA_OBJECT_INPUT
    RULE_SEQUENCE_UI = o9Constants.DP_RULE_SEQUENCE_INPUT
    REALIGNMENT_PERCENTAGE_UI = o9Constants.DP_REALIGNMENT_PERCENTAGE_INPUT
    CONVERSION_FACTOR_UI = o9Constants.DP_CONVERSION_FACTOR_INPUT
    TRANSITION_START_DATE_UI = o9Constants.TRANSITION_START_DATE_INPUT
    TRANSITION_END_DATE_UI = o9Constants.TRANSITION_END_DATE_INPUT
    HISTORY_ACTIVE_PERIODS_UI = o9Constants.HISTORY_REALIGNMENT_ACTIVE_PERIOD_INPUT

    # --- Attribute Mapping Columns ---
    DO_ACCOUNT = o9Constants.DATA_OBJECT_ACCOUNT_LEVEL
    DO_CHANNEL = o9Constants.DATA_OBJECT_CHANNEL_LEVEL
    DO_ITEM = o9Constants.DATA_OBJECT_ITEM_LEVEL
    DO_LOCATION = o9Constants.DATA_OBJECT_LOCATION_LEVEL
    DO_REGION = o9Constants.DATA_OBJECT_REGION_LEVEL
    DO_PNL = o9Constants.DATA_OBJECT_PNL_LEVEL
    DO_DEMAND_DOMAIN = o9Constants.DATA_OBJECT_DEMAND_DOMAIN_LEVEL

    REALIGN_HISTORY = o9Constants.REALIGN_HISTORY
    REALIGN_FORECAST = o9Constants.REALIGN_FORECAST

    TIME_DAY_KEY = o9Constants.DAY_KEY

    # --- Validation Output Columns ---
    VALIDATION_STATUS = o9Constants.VALIDATION_STATUS
    REJECT_FLAG_COL = o9Constants.REALIGNMENT_DATA_REJECT_FLAG
    VALIDATION_REMARK_COL = o9Constants.REALIGNMENT_DATA_VALIDATION_REMARK

    # --- ValidRulesOP Final Column Names ---
    FROM_ACCOUNT = o9Constants.DP_FROM_ACCOUNT_SCOPE
    FROM_CHANNEL = o9Constants.DP_FROM_CHANNEL_SCOPE
    FROM_DEMAND_DOMAIN = o9Constants.DP_FROM_DEMAND_DOMAIN_SCOPE
    FROM_ITEM = o9Constants.DP_FROM_ITEM_SCOPE
    FROM_LOCATION = o9Constants.DP_FROM_LOCATION_SCOPE
    FROM_PNL = o9Constants.DP_FROM_PNL_SCOPE
    FROM_REGION = o9Constants.DP_FROM_REGION_SCOPE

    TO_ACCOUNT = o9Constants.DP_TO_ACCOUNT_SCOPE
    TO_CHANNEL = o9Constants.DP_TO_CHANNEL_SCOPE
    TO_DEMAND_DOMAIN = o9Constants.DP_TO_DEMAND_DOMAIN_SCOPE
    TO_ITEM = o9Constants.DP_TO_ITEM_SCOPE
    TO_LOCATION = o9Constants.DP_TO_LOCATION_SCOPE
    TO_PNL = o9Constants.DP_TO_PNL_SCOPE
    TO_REGION = o9Constants.DP_TO_REGION_SCOPE

    UI_FROM_ACCOUNT = o9Constants.DP_FROM_ACCOUNT_SCOPE_UI
    UI_FROM_CHANNEL = o9Constants.DP_FROM_CHANNEL_SCOPE_UI
    UI_FROM_DEMAND_DOMAIN = o9Constants.DP_FROM_DEMAND_DOMAIN_SCOPE_UI
    UI_FROM_ITEM = o9Constants.DP_FROM_ITEM_SCOPE_UI
    UI_FROM_LOCATION = o9Constants.DP_FROM_LOCATION_SCOPE_UI
    UI_FROM_PNL = o9Constants.DP_FROM_PNL_SCOPE_UI
    UI_FROM_REGION = o9Constants.DP_FROM_REGION_SCOPE_UI

    UI_TO_ACCOUNT = o9Constants.DP_TO_ACCOUNT_SCOPE_UI
    UI_TO_CHANNEL = o9Constants.DP_TO_CHANNEL_SCOPE_UI
    UI_TO_DEMAND_DOMAIN = o9Constants.DP_TO_DEMAND_DOMAIN_SCOPE_UI
    UI_TO_ITEM = o9Constants.DP_TO_ITEM_SCOPE_UI
    UI_TO_LOCATION = o9Constants.DP_TO_LOCATION_SCOPE_UI
    UI_TO_PNL = o9Constants.DP_TO_PNL_SCOPE_UI
    UI_TO_REGION = o9Constants.DP_TO_REGION_SCOPE_UI

    RULE_SEQUENCE = o9Constants.DP_REALIGNMENT_RULE_SEQUENCE
    REALIGNMENT_PERCENTAGE = o9Constants.DP_REALIGNMENT_PERCENTAGE
    CONVERSION_FACTOR = o9Constants.DP_CONVERSION_FACTOR
    TRANSITION_START_DATE = o9Constants.TRANSITION_START_DATE
    TRANSITION_END_DATE = o9Constants.TRANSITION_END_DATE
    HISTORY_ACTIVE_PERIOD = o9Constants.HISTORY_REALIGNMENT_ACTIVE_PERIOD

    # --- Rules created measure ---
    RULE_CREATED_BY = o9Constants.DP_RULE_SEQUENCE_CREATED_BY
    RULE_CREATED_DATE = o9Constants.DP_RULE_SEQUENCE_CREATED_DATE

    # Internal Constants
    VALIDATION_STATUS_GOOD = "GOOD"
    VALIDATION_STATUS_BAD = "BAD"

    DEFAULT_NUM_RULES = 50

    # Measure for incremental realignment rules
    REALIGNMENT_RULE_ID = "RealignmentRuleID"

    UI_FROM_DIM = [
        UI_FROM_ACCOUNT,
        UI_FROM_CHANNEL,
        UI_FROM_DEMAND_DOMAIN,
        UI_FROM_ITEM,
        UI_FROM_LOCATION,
        UI_FROM_PNL,
        UI_FROM_REGION,
    ]

    UI_TO_DIM = [
        UI_TO_ACCOUNT,
        UI_TO_CHANNEL,
        UI_TO_DEMAND_DOMAIN,
        UI_TO_ITEM,
        UI_TO_LOCATION,
        UI_TO_PNL,
        UI_TO_REGION,
    ]

    # --- List of all required columns ---

    VALID_RULES_OP_MAP_COLS = [
        VERSION_COL,
        DATA_OBJECT,
        SEQUENCE_COL,
        FROM_ACCOUNT,
        FROM_CHANNEL,
        FROM_DEMAND_DOMAIN,
        FROM_ITEM,
        FROM_LOCATION,
        FROM_PNL,
        FROM_REGION,
        TO_ACCOUNT,
        TO_CHANNEL,
        TO_DEMAND_DOMAIN,
        TO_ITEM,
        TO_LOCATION,
        TO_PNL,
        TO_REGION,
        RULE_SEQUENCE,
        REALIGNMENT_PERCENTAGE,
        CONVERSION_FACTOR,
        TRANSITION_START_DATE,
        TRANSITION_END_DATE,
        HISTORY_ACTIVE_PERIOD,
    ]

    VALID_RULES_OP_FINAL_COLS = (
        VALID_RULES_OP_MAP_COLS + UI_FROM_DIM + UI_TO_DIM + [RULE_CREATED_BY, RULE_CREATED_DATE]
    )

    REMARKS_OP_FINAL_COLS = [
        VERSION_COL,
        REALIGNMENT_DATA_OBJECT_UI,
        SEQUENCE_COL,
        FROM_ACCOUNT_UI,
        FROM_CHANNEL_UI,
        FROM_DEMAND_DOMAIN_UI,
        FROM_ITEM_UI,
        FROM_LOCATION_UI,
        FROM_PNL_UI,
        FROM_REGION_UI,
        TO_ACCOUNT_UI,
        TO_CHANNEL_UI,
        TO_DEMAND_DOMAIN_UI,
        TO_ITEM_UI,
        TO_LOCATION_UI,
        TO_PNL_UI,
        TO_REGION_UI,
        RULE_SEQUENCE_UI,
        REALIGNMENT_PERCENTAGE_UI,
        CONVERSION_FACTOR_UI,
        TRANSITION_START_DATE_UI,
        TRANSITION_END_DATE_UI,
        HISTORY_ACTIVE_PERIODS_UI,
    ]

    # --- Mappings for validation ---
    DIMENSION_MAP = {
        "Account": {
            "from": FROM_ACCOUNT_UI,
            "to": TO_ACCOUNT_UI,
            "level": DO_ACCOUNT,
        },
        "Channel": {
            "from": FROM_CHANNEL_UI,
            "to": TO_CHANNEL_UI,
            "level": DO_CHANNEL,
        },
        "DemandDomain": {
            "from": FROM_DEMAND_DOMAIN_UI,
            "to": TO_DEMAND_DOMAIN_UI,
            "level": DO_DEMAND_DOMAIN,
        },
        "Item": {"from": FROM_ITEM_UI, "to": TO_ITEM_UI, "level": DO_ITEM},
        "Location": {
            "from": FROM_LOCATION_UI,
            "to": TO_LOCATION_UI,
            "level": DO_LOCATION,
        },
        "PnL": {"from": FROM_PNL_UI, "to": TO_PNL_UI, "level": DO_PNL},
        "Region": {
            "from": FROM_REGION_UI,
            "to": TO_REGION_UI,
            "level": DO_REGION,
        },
    }

    REMARKS_OP_GRAINS = [VERSION_COL, PERSONNEL_EMAIL, SEQUENCE_COL]
    VALID_RULES_OP_GRAINS = [
        VERSION_COL,
        DATA_OBJECT,
        DM_RULE,
        SEQUENCE_COL,
    ]


def append_validation_remark(df, mask, remark_col, new_remark):
    """Appends a new remark to the validation remark column for rows matching the mask."""
    try:
        # Ensure mask is a boolean Series aligned with df's index
        if not isinstance(mask, pd.Series):
            aligned_mask = pd.Series(False, index=df.index)
            aligned_mask.loc[mask] = True
            mask = aligned_mask

        if mask.any():
            current_remarks = df.loc[mask, remark_col].fillna("").astype(str)
            # Add separator only if the remark is not empty
            separator = np.where(current_remarks == "", "", "; ")
            df.loc[mask, remark_col] = current_remarks + separator + new_remark
    except Exception as e:
        logger.exception("Exception for append_validation_remark : {}".format(e))


def prepare_remarktable(file_upload_df, attribute_mapping_df, cfg):
    """
    Merges FileUpload with AttributeMapping and initializes validation columns.
    Returns a single DataFrame with all rows from FileUpload.
    """
    try:
        logger.info("Preparing remarktable by merging FileUpload with AttributeMapping")

        # Iterate through each dimension's dictionary in the map
        for dim_map in cfg.DIMENSION_MAP.values():
            # Get the column name for the 'level'
            level_col = dim_map["level"]

            if level_col in attribute_mapping_df.columns:
                attribute_mapping_df[level_col] = attribute_mapping_df[level_col].map(
                    remove_first_brackets
                )

        attribute_mapping_df = attribute_mapping_df.rename(
            columns={cfg.DATA_OBJECT: cfg.REALIGNMENT_DATA_OBJECT_UI}
        )

        remarktable = pd.merge(
            file_upload_df,
            attribute_mapping_df,
            on=[cfg.VERSION_COL, cfg.REALIGNMENT_DATA_OBJECT_UI],
            how="left",
        )

        remarktable[cfg.REJECT_FLAG_COL] = False
        remarktable[cfg.VALIDATION_REMARK_COL] = ""

        # 1. Initial Check: Data Object exists in Attribute Mapping
        no_match_mask = remarktable[cfg.REALIGNMENT_DATA_OBJECT_UI].isnull()
        append_validation_remark(
            remarktable,
            no_match_mask,
            cfg.VALIDATION_REMARK_COL,
            "Realignment Data Object not found in Attribute Mapping",
        )
        remarktable.loc[no_match_mask, cfg.REJECT_FLAG_COL] = True

        logger.info("Remarktable prepared with {} rows".format(len(remarktable)))

    except Exception as e:
        logger.exception("Exception for prepare_remarktable : {}".format(e))

    return remarktable


def validate_numeric_fields(remarktable, cfg):
    """Validates that specified fields are numeric and within range."""
    try:
        logger.info("Validating numeric fields in remarktable")
        base_mask = ~remarktable[cfg.REJECT_FLAG_COL]
        numeric_cols = {
            cfg.REALIGNMENT_PERCENTAGE_UI: 100,
            cfg.CONVERSION_FACTOR_UI: None,
        }
        for col, max_val in numeric_cols.items():
            # Check for empty values in required numeric fields
            if col == cfg.REALIGNMENT_PERCENTAGE_UI:
                empty_mask = base_mask & (
                    remarktable[col].isnull() | (remarktable[col].astype(str).str.strip() == "")
                )
                append_validation_remark(
                    remarktable,
                    empty_mask,
                    cfg.VALIDATION_REMARK_COL,
                    f"{col} cannot be empty.",
                )
                remarktable.loc[empty_mask, cfg.REJECT_FLAG_COL] = True
                base_mask &= ~empty_mask  # Exclude from subsequent checks in this loop

            # Check for invalid number format
            numeric_series = pd.to_numeric(remarktable[col], errors="coerce")
            nan_mask = (
                base_mask
                & numeric_series.isnull()
                & remarktable[col].notnull()
                & (remarktable[col].astype(str).str.strip() != "")
            )
            append_validation_remark(
                remarktable,
                nan_mask,
                cfg.VALIDATION_REMARK_COL,
                f"{col} must be a valid number.",
            )
            remarktable.loc[nan_mask, cfg.REJECT_FLAG_COL] = True
            base_mask &= ~nan_mask

            # Check if the value is within the specified range (e.g., 0-100)
            if max_val is not None:
                range_mask = base_mask & ((numeric_series < 0) | (numeric_series > max_val))
                append_validation_remark(
                    remarktable,
                    range_mask,
                    cfg.VALIDATION_REMARK_COL,
                    f"{col} must be between 0 and {max_val}.",
                )
                remarktable.loc[range_mask, cfg.REJECT_FLAG_COL] = True

        logger.info("Numeric fields validation completed")

    except Exception as e:
        logger.exception("Exception for validate_numeric_fields : {}".format(e))

    return remarktable


def validate_dates(remarktable, cfg, base_mask, current_day_key):
    """Validates transition dates for format and logic."""
    try:
        logger.info("Validating transition dates in remarktable")
        start_date = pd.to_datetime(remarktable[cfg.TRANSITION_START_DATE_UI], errors="coerce")
        end_date = pd.to_datetime(remarktable[cfg.TRANSITION_END_DATE_UI], errors="coerce")

        # Check for empty Transition Start Date
        start_empty_mask = base_mask & remarktable[cfg.TRANSITION_START_DATE_UI].isnull()
        append_validation_remark(
            remarktable,
            start_empty_mask,
            cfg.VALIDATION_REMARK_COL,
            "Transition Start Date cannot be empty.",
        )
        remarktable.loc[start_empty_mask, cfg.REJECT_FLAG_COL] = True

        # Check for invalid (but not empty) Transition Start Date format
        start_invalid_mask = (
            base_mask
            & ~start_empty_mask
            & start_date.isnull()
            & remarktable[cfg.TRANSITION_START_DATE_UI].notnull()
        )
        append_validation_remark(
            remarktable,
            start_invalid_mask,
            cfg.VALIDATION_REMARK_COL,
            "Transition Start Date is not a valid date.",
        )
        remarktable.loc[start_invalid_mask, cfg.REJECT_FLAG_COL] = True

        # Check that end date is not before start date
        date_logic_mask = (
            base_mask & (start_date > end_date) & start_date.notnull() & end_date.notnull()
        )
        append_validation_remark(
            remarktable,
            date_logic_mask,
            cfg.VALIDATION_REMARK_COL,
            "Transition End Date must be after Transition Start Date.",
        )
        remarktable.loc[date_logic_mask, cfg.REJECT_FLAG_COL] = True

        # Check for  rows that are purely for forecast realignment.
        # Create a mask for rows that are for forecast realignment only
        # and have not already been rejected for other date issues.
        forecast_only_mask = (
            base_mask
            & ~start_empty_mask
            & ~start_invalid_mask
            & (remarktable[cfg.REALIGN_FORECAST] == 1)
            & (remarktable[cfg.REALIGN_HISTORY] == 0)
        )

        # Within that set, find rows where the start date is before the current day.
        # This comparison requires that start_date is a valid date.
        invalid_start_date_mask = forecast_only_mask & (start_date < current_day_key)

        if invalid_start_date_mask.any():
            append_validation_remark(
                remarktable,
                invalid_start_date_mask,
                cfg.VALIDATION_REMARK_COL,
                "Cannot accept a Transition date in the past for Forecast realignment",
            )
            remarktable.loc[invalid_start_date_mask, cfg.REJECT_FLAG_COL] = True

        logger.info("Transition dates validation completed")

    except Exception as e:
        logger.exception("Exception for validate_dates : {}".format(e))

    return remarktable


def validate_scope_consistency(remarktable, cfg, base_mask):
    """
    Checks that data is only provided for scopes specified by the Data Object level columns.
    """
    try:
        logger.info("Validating scope consistency in remarktable")
        # 'allowed_dims_map' determines which dimensions are PERMITTED for each row
        # based on the populated 'Data Object ... Level' columns.
        allowed_dims_map = {
            dim: (remarktable[dim_map["level"]].notna() & (remarktable[dim_map["level"]] != ""))
            for dim, dim_map in cfg.DIMENSION_MAP.items()
        }

        # 'populated_masks' determines which dimension scopes have been FILLED IN for each row.
        populated_masks = {
            dim: {
                "from": remarktable[dim_map["from"]].notnull()
                & (remarktable[dim_map["from"]].astype(str).str.strip() != ""),
                "to": remarktable[dim_map["to"]].notnull()
                & (remarktable[dim_map["to"]].astype(str).str.strip() != ""),
            }
            for dim, dim_map in cfg.DIMENSION_MAP.items()
        }

        # --- Validation Step ---
        # Iterate through each dimension to find inconsistencies.
        for dim in cfg.DIMENSION_MAP.keys():
            is_populated = populated_masks[dim]["from"] | populated_masks[dim]["to"]
            is_allowed = allowed_dims_map[dim]

            # A row fails if a scope is populated when it is not allowed.
            consistency_fail_mask = base_mask & is_populated & ~is_allowed

            if consistency_fail_mask.any():
                append_validation_remark(
                    remarktable,
                    consistency_fail_mask,
                    cfg.VALIDATION_REMARK_COL,
                    "Dimension value provided for a scope not specified in the Realignment Data Object.",
                )
                remarktable.loc[consistency_fail_mask, cfg.REJECT_FLAG_COL] = True

        logger.info("Scope consistency validation completed")

    except Exception as e:
        logger.exception("Exception for validate_scope_consistency : {}".format(e))

    return remarktable


def validate_leaf_level_rules(remarktable, cfg, base_mask):
    """
    Validates rules related to leaf-level dimensions.
    1. For single-dimension rules, the dimension must be at its leaf level.
    2. For multi-dimension rules, only the leaf-level dimension can have different 'From'/'To' scopes.
    """
    try:
        logger.info("Validating leaf-level rules in remarktable")
        # Check for Realign History = 1 and Realign Forecast = 0 or None
        history_is_active_mask = remarktable[cfg.REALIGN_HISTORY] == 1.0
        forecast_is_inactive_mask = (remarktable[cfg.REALIGN_FORECAST] == 0.0) | (
            remarktable[cfg.REALIGN_FORECAST].isnull()
        )

        mask = history_is_active_mask & forecast_is_inactive_mask
        base_mask &= mask

        # Create a DataFrame to count populated dimension levels for each row
        level_cols = [dim_map["level"] for dim_map in cfg.DIMENSION_MAP.values()]
        populated_levels_df = remarktable[level_cols].notna() & (remarktable[level_cols] != "")
        num_populated_dims = populated_levels_df.sum(axis=1)

        # Mask for rows with exactly one dimension specified
        single_dim_mask = base_mask & (num_populated_dims == 1)

        # Mask for rows with more than one dimension specified
        multi_dim_mask = base_mask & (num_populated_dims > 1)

        # Iterate through each possible dimension to apply the rules
        for dim, dim_map in cfg.DIMENSION_MAP.items():
            from_col, to_col, level_col = (
                dim_map["from"],
                dim_map["to"],
                dim_map["level"],
            )

            # Define the leaf level attribute name (e.g., "Item.[Item]", "Demand Domain.[Demand Domain]")
            leaf_name_map = {"DemandDomain": "Demand Domain"}
            dim_name_for_leaf = leaf_name_map.get(dim, dim)
            leaf_level = f"{dim_name_for_leaf}.[{dim_name_for_leaf}]"

            # Continue if this dimension is not used in any remaining rows
            if not (remarktable.loc[base_mask, level_col].notna().any()):
                continue

            is_this_dim_active = populated_levels_df[level_col]
            is_not_leaf = remarktable[level_col] != leaf_level

            # --- Rule 1: Single-dimension rules must be at leaf level ---
            v1_fail_mask = single_dim_mask & is_this_dim_active & is_not_leaf
            if v1_fail_mask.any():
                append_validation_remark(
                    remarktable,
                    v1_fail_mask,
                    cfg.VALIDATION_REMARK_COL,
                    f"For history realignment single-dimension realignment for {dim} must be at leaf level ({leaf_level}).",
                )
                remarktable.loc[v1_fail_mask, cfg.REJECT_FLAG_COL] = True

            # --- Rule 2: In multi-dim rules, non-leaf scopes must be identical ---
            # Using different fillna values ensures that two NA/nulls are not considered equal
            scopes_differ = remarktable[from_col].fillna("val_A") != remarktable[to_col].fillna(
                "val_B"
            )
            v2_fail_mask = multi_dim_mask & is_this_dim_active & is_not_leaf & scopes_differ

            if v2_fail_mask.any():
                append_validation_remark(
                    remarktable,
                    v2_fail_mask,
                    cfg.VALIDATION_REMARK_COL,
                    f"For history realignment From/To scopes for {dim} must be identical as it is a non-leaf dimension.",
                )
                remarktable.loc[v2_fail_mask, cfg.REJECT_FLAG_COL] = True

        logger.info("Leaf-level rules validation completed")

    except Exception as e:
        logger.exception("Exception for validate_leaf_level_rules : {}".format(e))

    return remarktable


def validate_dimension_scopes(remarktable, cfg, base_mask, master_data_map):
    """
    Performs mandatory, consistency, and master data checks.
    """
    try:
        logger.info("Validating dimension scopes in remarktable")
        # Mandatory and Master Data Checks ---
        for dim, dim_map in cfg.DIMENSION_MAP.items():
            master_df = master_data_map.get(dim)
            from_col, to_col, level_col = (
                dim_map["from"],
                dim_map["to"],
                dim_map["level"],
            )

            required_dim_mask = (
                base_mask & remarktable[level_col].notna() & (remarktable[level_col] != "")
            )
            if not required_dim_mask.any():
                continue

            # A. Mandatory Check: 'From' AND 'To' must be provided for the required dimension.
            from_missing = remarktable[from_col].isnull() | (
                remarktable[from_col].astype(str).str.strip() == ""
            )
            to_missing = remarktable[to_col].isnull() | (
                remarktable[to_col].astype(str).str.strip() == ""
            )
            mandatory_fail_mask = required_dim_mask & (from_missing | to_missing)
            append_validation_remark(
                remarktable,
                mandatory_fail_mask,
                cfg.VALIDATION_REMARK_COL,
                f"{dim} members are mandatory for this realignment type.",
            )
            remarktable.loc[mandatory_fail_mask, cfg.REJECT_FLAG_COL] = True

            dim_check_mask = required_dim_mask & ~mandatory_fail_mask
            if master_df is None or not dim_check_mask.any():
                continue

            # B. Master Data Member Validation (Optimized)
            active_levels = remarktable.loc[dim_check_mask, level_col].dropna().unique()

            for level in active_levels:
                level_group_mask = dim_check_mask & (remarktable[level_col] == level)

                if not level or level not in master_df.columns:
                    append_validation_remark(
                        remarktable,
                        level_group_mask,
                        cfg.VALIDATION_REMARK_COL,
                        f"Invalid validation level '{level}' for {dim} Master.",
                    )
                    remarktable.loc[level_group_mask, cfg.REJECT_FLAG_COL] = True
                    continue

                # Construct UI-level column name and get valid members
                ui_level = level
                if level.endswith("]"):
                    ui_level_candidate = level[:-1] + "$DisplayName]"
                    if ui_level_candidate in master_df.columns:
                        ui_level = ui_level_candidate

                valid_ui_members = set(master_df[ui_level].dropna().astype(str))
                valid_members = set(master_df[level].dropna().astype(str))

                valid_members.update(valid_ui_members)

                # Validate 'from' and 'to' members using vectorized .isin()
                from_invalid_mask = level_group_mask & ~remarktable[from_col].isin(valid_members)
                append_validation_remark(
                    remarktable,
                    from_invalid_mask,
                    cfg.VALIDATION_REMARK_COL,
                    f"{dim} Member not found in {dim} Master.",
                )
                remarktable.loc[from_invalid_mask, cfg.REJECT_FLAG_COL] = True

                to_invalid_mask = level_group_mask & ~remarktable[to_col].isin(valid_members)
                append_validation_remark(
                    remarktable,
                    to_invalid_mask,
                    cfg.VALIDATION_REMARK_COL,
                    f"{dim} Member not found in {dim} Master.",
                )
                remarktable.loc[to_invalid_mask, cfg.REJECT_FLAG_COL] = True

        logger.info("Dimension scopes validation completed")

    except Exception as e:
        logger.exception("Exception for validate_dimension_scopes : {}".format(e))

    return remarktable


def validate_from_to_equality(remarktable, cfg, base_mask):
    """
    Checks if all populated 'from' scope values are identical to their 'to' scope counterparts.
    """
    try:
        logger.info("Validating From/To equality in remarktable")
        # Create a dictionary of boolean Series for each dimension to track population status.
        is_populated_series = {
            dim: (
                remarktable[dim_map["from"]].notna()
                & (remarktable[dim_map["from"]].astype(str).str.strip() != "")
                & remarktable[dim_map["to"]].notna()
                & (remarktable[dim_map["to"]].astype(str).str.strip() != "")
            )
            for dim, dim_map in cfg.DIMENSION_MAP.items()
        }

        # Create a dictionary of boolean Series to track if 'from' and 'to' values are equal.
        # Using different fillna values ensures that two NA/nulls are not considered equal.
        are_equal_series = {
            dim: (
                remarktable[dim_map["from"]].fillna("value_A")
                == remarktable[dim_map["to"]].fillna("value_B")
            )
            for dim, dim_map in cfg.DIMENSION_MAP.items()
        }

        # Convert dictionaries of Series to DataFrames for row-wise analysis.
        df_populated = pd.DataFrame(is_populated_series)
        df_equal = pd.DataFrame(are_equal_series)

        # A row fails if:
        # 1. For every dimension, it's either true that (from == to) OR the dimension is not populated.
        #    This is equivalent to: for all POPULATED dimensions, from == to.
        all_populated_are_equal = (df_equal | ~df_populated).all(axis=1)

        # 2. At least one dimension scope is actually populated for the row.
        any_populated = df_populated.any(axis=1)

        # Combine the conditions with the base_mask to identify failing rows.
        equality_fail_mask = base_mask & all_populated_are_equal & any_populated

        if equality_fail_mask.any():
            append_validation_remark(
                remarktable,
                equality_fail_mask,
                cfg.VALIDATION_REMARK_COL,
                "All 'From' and 'To' scope values are identical.",
            )
            remarktable.loc[equality_fail_mask, cfg.REJECT_FLAG_COL] = True

        logger.info("From/To equality validation completed")

    except Exception as e:
        logger.exception("Exception for validate_from_to_equality : {}".format(e))

    return remarktable


def validate_duplicates(remarktable, cfg, base_mask):
    """
    Checks for duplicate rows based on all scope columns and the transition start date.
    """
    try:
        logger.info("Validating duplicate rows in remarktable")
        # Create a flat list of all 'from' and 'to' scope columns.
        duplicate_check_cols = []
        for dim_map in cfg.DIMENSION_MAP.values():
            duplicate_check_cols.append(dim_map["from"])
            duplicate_check_cols.append(dim_map["to"])

        # Add the transition start date to the list of columns to check.
        duplicate_check_cols.append(cfg.TRANSITION_START_DATE_UI)
        duplicate_check_cols.append(cfg.REALIGNMENT_DATA_OBJECT_UI)

        # Identify duplicates, keeping the first occurrence and flagging the rest.
        duplicate_mask = base_mask & remarktable.duplicated(
            subset=duplicate_check_cols, keep="first"
        )

        # Append remark and set reject flag for the identified duplicate rows.
        append_validation_remark(
            remarktable,
            duplicate_mask,
            cfg.VALIDATION_REMARK_COL,
            "Duplicate row",
        )
        remarktable.loc[duplicate_mask, cfg.REJECT_FLAG_COL] = True

        logger.info("Duplicate rows validation completed")

    except Exception as e:
        logger.exception("Exception for validate_duplicates : {}".format(e))

    return remarktable


def run_all_validations(remarktable, cfg, master_data_map, current_day_key):
    """
    All validation checks by calling specific sub-functions.
    """
    try:
        logger.info("Running all validations on remarktable")

        # 1. Numeric Field Validation
        remarktable = validate_numeric_fields(remarktable, cfg)

        # Base mask to exclude rows already rejected from subsequent checks
        base_mask = ~remarktable[cfg.REJECT_FLAG_COL]

        # 2. Date Validation
        remarktable = validate_dates(remarktable, cfg, base_mask, current_day_key)
        base_mask = ~remarktable[cfg.REJECT_FLAG_COL]

        # 3. Scope Consistency Validation
        remarktable = validate_scope_consistency(remarktable, cfg, base_mask)
        base_mask = ~remarktable[cfg.REJECT_FLAG_COL]

        # 4. Leaf-Level Logic Validation
        remarktable = validate_leaf_level_rules(remarktable, cfg, base_mask)
        base_mask = ~remarktable[cfg.REJECT_FLAG_COL]

        # 5. Dimension-Based Validations
        remarktable = validate_dimension_scopes(remarktable, cfg, base_mask, master_data_map)
        base_mask = ~remarktable[cfg.REJECT_FLAG_COL]

        # 6. From/To Equality Check
        remarktable = validate_from_to_equality(remarktable, cfg, base_mask)
        base_mask = ~remarktable[cfg.REJECT_FLAG_COL]

        # 7. Duplicate Row Validation
        remarktable = validate_duplicates(remarktable, cfg, base_mask)

        logger.info("All validations completed successfully")

    except Exception as e:
        logger.exception("Exception for run_all_validations : {}".format(e))

    return remarktable


def transform_good_records(remarktable, master_data_map, cfg):
    """
    For records with "GOOD" validation status, this function:
    1.  Ensures the original 'from' and 'to' columns contain the canonical system names.
    2.  Adds new UI-specific columns and populates them with the corresponding display names.
    3.  Fills empty 'History Realignment Active Period Input' with a default value of 90.
    """
    try:
        logger.info("Transforming good records in remarktable")
        # Create a mask to identify rows that passed all validations.
        good_mask = remarktable[cfg.VALIDATION_STATUS] == Constants.VALIDATION_STATUS_GOOD

        # --- Add new UI columns to the DataFrame and initialize them ---
        # This is done upfront to ensure the columns exist before we populate them.
        for col in cfg.UI_FROM_DIM + cfg.UI_TO_DIM:
            if col not in remarktable.columns:
                remarktable[col] = np.nan

        # --- Transform columns to system names and populate new UI display name columns ---
        # We zip the dimension map with the new UI column names.
        # This assumes the order in cfg.DIMENSION_MAP matches the order in cfg.UI_FROM_DIM/UI_TO_DIM.
        dim_items = list(cfg.DIMENSION_MAP.items())
        for i, (dim, dim_map) in enumerate(dim_items):
            master_df = master_data_map.get(dim)
            if master_df is None:
                continue

            # Get column names for the current dimension
            from_col, to_col, level_col = (
                dim_map["from"],
                dim_map["to"],
                dim_map["level"],
            )
            ui_from_col, ui_to_col = cfg.UI_FROM_DIM[i], cfg.UI_TO_DIM[i]

            active_levels = remarktable.loc[good_mask, level_col].dropna().unique()

            for level in active_levels:
                # Determine the display name column from the master data
                ui_level = level
                if level.endswith("]"):
                    ui_level_candidate = level[:-1] + "$DisplayName]"
                    if ui_level_candidate in master_df.columns:
                        ui_level = ui_level_candidate

                if (
                    level not in master_df.columns
                    or ui_level not in master_df.columns
                    or level == ui_level
                ):
                    continue

                # --- Create Mappings ---

                # 1. Map to get SYSTEM names (handles both display or system name as input)
                system_to_system_map = pd.Series(
                    master_df[level].values, index=master_df[level].values
                )
                display_to_system_map = pd.Series(
                    master_df[level].values, index=master_df[ui_level].values
                )
                map_to_system = {
                    **system_to_system_map.dropna().to_dict(),
                    **display_to_system_map.dropna().to_dict(),
                }

                # 2. Map to get DISPLAY names (handles both display or system name as input)
                system_to_display_map = pd.Series(
                    master_df[ui_level].values, index=master_df[level].values
                )
                display_to_display_map = pd.Series(
                    master_df[ui_level].values,
                    index=master_df[ui_level].values,
                )
                map_to_display = {
                    **display_to_display_map.dropna().to_dict(),
                    **system_to_display_map.dropna().to_dict(),
                }

                if not map_to_system:  # If one map is empty, the other will be too
                    continue

                # --- Apply Mappings ---
                level_mask = good_mask & (remarktable[level_col] == level)

                # First, populate the new UI columns with DISPLAY names.
                # We use the original 'from'/'to' columns as the source for the mapping.
                remarktable.loc[level_mask, ui_from_col] = remarktable.loc[
                    level_mask, from_col
                ].replace(map_to_display)
                remarktable.loc[level_mask, ui_to_col] = remarktable.loc[
                    level_mask, to_col
                ].replace(map_to_display)

                # Then, update the original columns to ensure they contain SYSTEM names.
                remarktable.loc[level_mask, from_col] = remarktable.loc[
                    level_mask, from_col
                ].replace(map_to_system)
                remarktable.loc[level_mask, to_col] = remarktable.loc[level_mask, to_col].replace(
                    map_to_system
                )

        # --- Default 'History Realignment Active Period Input' for good records ---
        history_periods_col = cfg.HISTORY_ACTIVE_PERIODS_UI
        if history_periods_col in remarktable.columns:
            is_empty_mask = remarktable[history_periods_col].isnull() | (
                remarktable[history_periods_col].astype(str).str.strip() == ""
            )
            fill_default_mask = good_mask & is_empty_mask
            remarktable.loc[fill_default_mask, history_periods_col] = 90

        logger.info("Good records transformed successfully")

    except Exception as e:
        logger.exception("Exception for transform_good_records : {}".format(e))

    return remarktable


def fetch_rule_ids(num_rules=Constants.DEFAULT_NUM_RULES):
    """Fetch unique Rule IDs using web API"""
    try:
        logger.info("Fetching {} Rule IDs using web API".format(num_rules))

        # Initialize O9API client like in member_creation.py
        tenant_api_client = O9API(verify=False, tenant_url="")
        query = f"select NextLabel([{Constants.REALIGNMENT_RULE_ID}]);"

        # First, attempt to authenticate. If neither method works, raise an exception.
        if tenant_api_client.initialize_session():
            logger.info("Session initialized successfully for Rule ID fetching")
        elif tenant_api_client.set_api_key():
            logger.warning("Retry with API KEY successful for Rule ID fetching")
        else:
            logger.error("Authentication unsuccessful for Rule ID fetching")
            raise Exception("Failed to authenticate for Rule ID generation")

        labels_set = set()
        execution_count = 0
        start_time = time.time()

        while len(labels_set) < num_rules:
            result = tenant_api_client.run_ibpl_query(query)
            execution_count += 1

            # Extract the actual value from the result
            if isinstance(result, dict):
                # The result is a dictionary, extract the actual value
                # Usually the first (and only) value in the result
                actual_value = list(result.values())[0] if result else None
            else:
                actual_value = result

            if actual_value:
                labels_set.add(actual_value)

        logger.info(
            "Query executed {} times in {:.2f} seconds".format(
                execution_count, time.time() - start_time
            )
        )
        logger.info("Final set of Rule IDs collected: {}".format(sorted(labels_set)))
        logger.info("Total unique Rule IDs generated: {}".format(len(labels_set)))
    except Exception as e:
        logger.exception("Exception for fetch_rule_ids : {}".format(e))
        return pd.DataFrame(columns=[Constants.DM_RULE])

    return pd.DataFrame(sorted(labels_set), columns=[Constants.DM_RULE])


@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(
    AttributeMapping,
    FileUpload,
    AccountMapping,
    ChannelMapping,
    PnLMapping,
    DemandDomainMapping,
    LocationMapping,
    ItemMapping,
    RegionMapping,
    CurrentDay,
    df_keys,
):
    """
    Validates Realignment Rules from FileUpload, returning all records in RemarksOP
    and only good records in ValidRulesOP.
    """
    plugin_name = "DP088ValidateRealignmentRulesFromFileUpload"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    cfg = Constants()

    cols_req_in_remarks_op = Constants.REMARKS_OP_GRAINS + [
        cfg.REJECT_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
    ]

    valid_rules_op_measures = [
        col for col in cfg.VALID_RULES_OP_FINAL_COLS if col not in Constants.VALID_RULES_OP_GRAINS
    ]
    cols_req_in_valid_rules_op = Constants.VALID_RULES_OP_GRAINS + valid_rules_op_measures

    # Initialize both outputs with the required schema
    RemarksOP = pd.DataFrame(columns=cols_req_in_remarks_op)
    ValidRulesOP = pd.DataFrame(columns=cols_req_in_valid_rules_op)

    try:
        dataframes_to_check = {
            "FileUpload": FileUpload,
            "AttributeMapping": AttributeMapping,
            "AccountMapping": AccountMapping,
            "ChannelMapping": ChannelMapping,
            "PnLMapping": PnLMapping,
            "DemandDomainMapping": DemandDomainMapping,
            "LocationMapping": LocationMapping,
            "ItemMapping": ItemMapping,
            "RegionMapping": RegionMapping,
            "CurrentDay": CurrentDay,
        }

        empty_dfs = [name for name, df in dataframes_to_check.items() if df.empty]
        if empty_dfs:
            error_message = f"Required DataFrames are empty:: {', '.join(empty_dfs)}"
            raise InputEmptyException(error_message)

        current_day_key = pd.to_datetime(CurrentDay[cfg.TIME_DAY_KEY].iloc[0])
        remarktable = prepare_remarktable(FileUpload, AttributeMapping, cfg)

        master_data_map = {
            "Item": ItemMapping,
            "Location": LocationMapping,
            "Channel": ChannelMapping,
            "Account": AccountMapping,
            "PnL": PnLMapping,
            "DemandDomain": DemandDomainMapping,
            "Region": RegionMapping,
        }

        remarktable = run_all_validations(remarktable, cfg, master_data_map, current_day_key)

        # Create a separate DataFrame for valid rule processing
        valid_rules_df = remarktable.copy()

        valid_rules_df[cfg.VALIDATION_STATUS] = np.where(
            valid_rules_df[cfg.REJECT_FLAG_COL],
            Constants.VALIDATION_STATUS_BAD,
            Constants.VALIDATION_STATUS_GOOD,
        )
        valid_rules_df = transform_good_records(valid_rules_df, master_data_map, cfg)

        # Finalize RemarksOP with all records
        for col in cols_req_in_remarks_op:
            if col not in remarktable.columns:
                remarktable[col] = pd.NA

        RemarksOP = remarktable[cols_req_in_remarks_op]

        # Create ValidRulesOP with only good records
        good_records_mask = ~valid_rules_df[cfg.REJECT_FLAG_COL]
        ValidRulesOP = valid_rules_df[good_records_mask].copy()
        ValidRulesOP[cfg.SEQUENCE_COL] = 1

        ValidRulesOP[cfg.RULE_CREATED_BY] = ValidRulesOP[cfg.PERSONNEL_EMAIL]
        ValidRulesOP[cfg.RULE_CREATED_DATE] = current_day_key

        rename_map = dict(zip(cfg.REMARKS_OP_FINAL_COLS, cfg.VALID_RULES_OP_MAP_COLS))

        for col in cfg.REMARKS_OP_FINAL_COLS:
            if col not in ValidRulesOP.columns:
                ValidRulesOP[col] = pd.NA

        ValidRulesOP = ValidRulesOP.rename(columns=rename_map)
        ValidRulesOP[cfg.DM_RULE] = pd.NA
        # fetch new rules IDs for good records
        rule_ids_needed = len(ValidRulesOP)
        if rule_ids_needed > 0:
            rule_ids_df = fetch_rule_ids(rule_ids_needed)
            ValidRulesOP[cfg.DM_RULE] = rule_ids_df[cfg.DM_RULE].values

            # upload rule IDs to o9 platform
            create_members(dataframe=rule_ids_df, logger=logger)

        ValidRulesOP = ValidRulesOP[cols_req_in_valid_rules_op]

        logger.info("Successfully executed {} ...".format(plugin_name))

    except InputEmptyException as e:
        logger.warning(f"Input Empty Exception occurred\n{e}")
        return RemarksOP, ValidRulesOP
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        return pd.DataFrame(columns=cols_req_in_remarks_op), pd.DataFrame(
            columns=cols_req_in_valid_rules_op
        )

    return RemarksOP, ValidRulesOP
