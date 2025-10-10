"""This is a helpers module for validating NPI assortment data from file uploads."""

import logging
import time

import pandas as pd
from o9Reference.API.member_creation import create_members
from o9Reference.API.o9api import O9API
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.date_utils import (
    convert_and_format_date,
    convert_and_validate_date_for_initiative,
    format_initiative_date,
    format_stage_date_to_mmddyyyy_with_time,
    get_current_date_midnight,
    is_valid_date,
)
from helpers.o9Constants import o9Constants

# Configure pandas
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None
pd.set_option("display.width", 1000)

logger = logging.getLogger("o9_logger")


# ===============================================================================
# CONFIGURATION
# ===============================================================================


class ValidationConfig:
    """Configuration for all column names and constant."""

    # Core Data Columns - Using existing constants from o9Constants
    VERSION_COL = o9Constants.VERSION_NAME
    PLANNING_ITEM_COL = o9Constants.PLANNING_ITEM
    PLANNING_CHANNEL_COL = o9Constants.PLANNING_CHANNEL
    PLANNING_ACCOUNT_COL = o9Constants.PLANNING_ACCOUNT
    PLANNING_PNL_COL = o9Constants.PLANNING_PNL
    PLANNING_DEMAND_DOMAIN_COL = o9Constants.PLANNING_DEMAND_DOMAIN
    PLANNING_REGION_COL = o9Constants.PLANNING_REGION
    PLANNING_LOCATION_COL = o9Constants.PLANNING_LOCATION

    # User Input Columns
    EMAIL_COL = o9Constants.EMAIL
    SEQUENCE_COL = o9Constants.SEQUENCE_COL
    INITIATIVE_NAME_INPUT_COL = o9Constants.NPI_INITIATIVE_NAME
    INITIATIVE_DESC_COL = o9Constants.NPI_INITIATIVE_DESCRIPTION
    LEVEL_NAME_COL = o9Constants.NPI_LEVEL_NAME
    ASSORTMENT_GROUP_COL = o9Constants.NPI_ASSORTMENT_GROUP_NUMBER
    DATA_OBJECT_COL = o9Constants.DATA_OBJECT

    # Entity Columns - NPI specific
    NPI_ITEM_COL = o9Constants.NPI_ITEM_NAME
    NPI_ACCOUNT_COL = o9Constants.NPI_ACCOUNT_NAME
    NPI_CHANNEL_COL = o9Constants.NPI_CHANNEL_NAME
    NPI_REGION_COL = o9Constants.NPI_REGION_NAME
    NPI_PNL_COL = o9Constants.NPI_PNL_NAME
    NPI_DEMAND_DOMAIN_COL = o9Constants.NPI_DEMAND_DOMAIN_NAME
    NPI_LOCATION_COL = o9Constants.NPI_LOCATION_NAME

    NPI_ITEM_LEVEL_COL = o9Constants.NPI_ITEM_LEVEL
    NPI_ACCOUNT_LEVEL_COL = o9Constants.NPI_ACCOUNT_LEVEL
    NPI_CHANNEL_LEVEL_COL = o9Constants.NPI_CHANNEL_LEVEL
    NPI_REGION_LEVEL_COL = o9Constants.NPI_REGION_LEVEL
    NPI_PNL_LEVEL_COL = o9Constants.NPI_PNL_LEVEL
    NPI_DEMAND_DOMAIN_LEVEL_COL = o9Constants.NPI_DEMAND_DOMAIN_LEVEL
    NPI_LOCATION_LEVEL_COL = o9Constants.NPI_LOCATION_LEVEL

    # Valid entries output columns
    VALID_ENTRIES_NPI_ITEM_COL = o9Constants.ITEM_NPI_ITEM
    VALID_ENTRIES_NPI_ACCOUNT_COL = o9Constants.ACCOUNT_NPI_ACCOUNT
    VALID_ENTRIES_NPI_CHANNEL_COL = o9Constants.CHANNEL_NPI_CHANNEL
    VALID_ENTRIES_NPI_REGION_COL = o9Constants.REGION_NPI_REGION
    VALID_ENTRIES_NPI_PNL_COL = o9Constants.PNL_NPI_PNL
    VALID_ENTRIES_NPI_DEMAND_DOMAIN_COL = o9Constants.DEMAND_DOMAIN_NPI_DEMAND_DOMAIN
    VALID_ENTRIES_NPI_LOCATION_COL = o9Constants.LOCATION_NPI_LOCATION

    # Date Columns
    NPI_LAUNCH_DATE_COL = o9Constants.NPI_PRODUCT_LAUNCH_DATE
    NPI_LAUNCH_DATE_L0_COL = o9Constants.NPI_PRODUCT_LAUNCH_DATE_L0
    NPI_EOL_DATE_COL = o9Constants.NPI_PRODUCT_EOL_DATE
    NPI_EOL_DATE_L0_COL = o9Constants.NPI_PRODUCT_EOL_DATE_L0
    NPI_INITIATIVE_START_DATE_COL = o9Constants.NPI_INITIATIVE_START_DATE
    NPI_INITIATIVE_END_DATE_COL = o9Constants.NPI_INITIATIVE_END_DATE
    LAUNCH_DATE_COL = o9Constants.PRODUCT_LAUNCH_DATE
    EOL_DATE_COL = o9Constants.PRODUCT_EOL_DATE

    # Validation Columns
    REJECTED_FLAG_COL = o9Constants.REJECTED_FLAG_COL
    VALIDATION_REMARK_COL = o9Constants.VALIDATION_REMARK_COL

    # Initiative columns
    INITIATIVE_CATEGORY_COL = o9Constants.INITIATIVE_CATEGORY
    INITIATIVE_TYPE_COL = o9Constants.INITIATIVE_TYPE
    INITIATIVE_DESCRIPTION_COL = o9Constants.INITIATIVE_DESCRIPTION
    INITIATIVE_TYPE_DISPLAY_NAME_COL = o9Constants.INITIATIVE_TYPE_DISPLAY_NAME
    INITIATIVE_DISPLAY_NAME_COL = o9Constants.INITIATIVE_DISPLAY_NAME
    INITIATIVE_NAME_OUTPUT_COL = o9Constants.INITIATIVE
    INITIATIVE_LEVEL_1_COL = o9Constants.INITIATIVE_LEVEL_1
    INITIATIVE_LEVEL_2_COL = o9Constants.INITIATIVE_LEVEL_2
    INITIATIVE_LEVEL_3_COL = o9Constants.INITIATIVE_LEVEL_3
    INITIATIVE_LEVEL_4_COL = o9Constants.INITIATIVE_LEVEL_4
    INITIATIVE_LEVEL_5_COL = o9Constants.INITIATIVE_LEVEL_5
    INITIATIVE_LEVEL_6_COL = o9Constants.INITIATIVE_LEVEL_6
    INITIATIVE_LEVEL_7_COL = o9Constants.INITIATIVE_LEVEL_7
    INITIATIVE_LEVEL_8_COL = o9Constants.INITIATIVE_LEVEL_8

    # Status columns
    NPI_INITIATIVE_STATUS_COL = o9Constants.NPI_INITIATIVE_STATUS
    INITIATIVE_OWNER_COL = o9Constants.INITIATIVE_OWNER
    NPI_INITIATIVE_CREATED_BY_COL = o9Constants.INITIATIVE_CREATED_BY
    NPI_INITIATIVE_CREATED_DATE_COL = o9Constants.INITIATIVE_CREATED_DATE
    NPI_INITIATIVE_LEVEL_ASSOCIATION_COL = o9Constants.NPI_INITIATIVE_LEVEL_ASSOCIATION
    NPI_Inititative_level_status_COL = o9Constants.NPI_INITIATIVE_LEVEL_STATUS
    NPI_Initiative_lowest_level_COL = o9Constants.NPI_INITIATIVE_LOWEST_LEVEL
    NPI_LEVEL_SEQUENCE_COL_L1 = o9Constants.NPI_LEVEL_SEQUENCE_L1
    NPI_fcst_generation_method_l1_COL = o9Constants.NPI_FORECAST_GENERATION_METHOD
    NPI_Ramp_up_bucket_l1_COL = o9Constants.NPI_RAMP_UP_BUCKET
    NPI_Ramp_up_period_l1_COL = o9Constants.NPI_RAMP_UP_PERIOD
    NPI_DATA_ACCEPTED_FLAG_COL = o9Constants.NPI_DATA_ACCEPTED_FLAG

    # Stage columns
    Data_validation_COL = o9Constants.DATA_VALIDATION
    Stage_association_COL = o9Constants.STAGE_ASSOCIATION
    Stage_owner_COL = o9Constants.STAGE_OWNER
    Stage_created_date_COL = o9Constants.STAGE_CREATED_DATE

    # Master Data Column Names
    MASTER_DATA_COLS = {
        "npi_level": o9Constants.DATA_OBJECT,
        "initiative": INITIATIVE_NAME_OUTPUT_COL,  # Use the output column reference
        "initiative_level_association": o9Constants.NPI_INITIATIVE_LEVEL_ASSOCIATION,
        "initiative_level_status": o9Constants.NPI_INITIATIVE_LEVEL_STATUS,
        "initiative_status": o9Constants.NPI_INITIATIVE_STATUS,
        "npi_association": o9Constants.NPI_ASSOCIATION_L0,
    }

    # Initiative Status Configuration
    INVALID_INITIATIVE_STATUSES = [
        "completed",
        "inactive",
    ]  # Invalid statuses that should be rejected

    npi_level_sequence = "NPI Level Sequence"
    # entity names
    Item_name = "Item"
    Account_name = "Account"
    Channel_name = "Channel"
    Region_name = "Region"
    PnL_name = "PnL"
    DemandDomain_name = "DemandDomain"
    Location_name = "Location"

    # Global NPI Levels
    global_npi_item_level = o9Constants.GLOBAL_NPI_ITEM_LEVEL
    global_npi_account_level = o9Constants.GLOBAL_NPI_ACCOUNT_LEVEL
    global_npi_channel_level = o9Constants.GLOBAL_NPI_CHANNEL_LEVEL
    global_npi_region_level = o9Constants.GLOBAL_NPI_REGION_LEVEL
    global_npi_pnl_level = o9Constants.GLOBAL_NPI_PNL_LEVEL
    global_npi_demand_domain_level = o9Constants.GLOBAL_NPI_DEMAND_DOMAIN_LEVEL
    global_npi_location_level = o9Constants.GLOBAL_NPI_LOCATION_LEVEL

    # Entity Master Data Mapping
    ENTITY_MAPPINGS = {
        NPI_ITEM_COL: "Item",
        NPI_ACCOUNT_COL: "Account",
        NPI_CHANNEL_COL: "Channel",
        NPI_REGION_COL: "Region",
        NPI_PNL_COL: "PnL",
        NPI_DEMAND_DOMAIN_COL: "DemandDomain",
        NPI_LOCATION_COL: "Location",
    }

    # Output Table required columns
    invalid_entries_output_req_cols = [
        VERSION_COL,
        EMAIL_COL,
        SEQUENCE_COL,
        REJECTED_FLAG_COL,
        VALIDATION_REMARK_COL,
    ]
    initiative_dim_data_req_output_cols = [
        INITIATIVE_NAME_OUTPUT_COL,
        INITIATIVE_DISPLAY_NAME_COL,
        INITIATIVE_DESCRIPTION_COL,
        INITIATIVE_TYPE_COL,
        INITIATIVE_LEVEL_1_COL,
        INITIATIVE_LEVEL_2_COL,
        INITIATIVE_LEVEL_3_COL,
        INITIATIVE_LEVEL_4_COL,
        INITIATIVE_LEVEL_5_COL,
        INITIATIVE_LEVEL_6_COL,
        INITIATIVE_LEVEL_7_COL,
        INITIATIVE_LEVEL_8_COL,
        INITIATIVE_CATEGORY_COL,
    ]
    initiative_fact_data_req_output_cols = [
        VERSION_COL,
        INITIATIVE_NAME_OUTPUT_COL,
        NPI_INITIATIVE_START_DATE_COL,
        NPI_INITIATIVE_END_DATE_COL,
        LAUNCH_DATE_COL,
        EOL_DATE_COL,
        NPI_INITIATIVE_STATUS_COL,
        NPI_INITIATIVE_CREATED_BY_COL,
        INITIATIVE_OWNER_COL,
        NPI_INITIATIVE_CREATED_DATE_COL,
    ]
    initiative_level_fact_data_req_output_cols = [
        VERSION_COL,
        INITIATIVE_NAME_OUTPUT_COL,
        DATA_OBJECT_COL,
        NPI_INITIATIVE_LEVEL_ASSOCIATION_COL,
        NPI_ITEM_LEVEL_COL,
        NPI_ACCOUNT_LEVEL_COL,
        NPI_CHANNEL_LEVEL_COL,
        NPI_REGION_LEVEL_COL,
        NPI_PNL_LEVEL_COL,
        NPI_DEMAND_DOMAIN_LEVEL_COL,
        NPI_LOCATION_LEVEL_COL,
        NPI_Inititative_level_status_COL,
        NPI_Initiative_lowest_level_COL,
        NPI_LEVEL_SEQUENCE_COL_L1,
        NPI_fcst_generation_method_l1_COL,
        NPI_Ramp_up_bucket_l1_COL,
        NPI_Ramp_up_period_l1_COL,
    ]
    valid_entries_output_req_cols = [
        VERSION_COL,
        EMAIL_COL,
        INITIATIVE_NAME_OUTPUT_COL,
        DATA_OBJECT_COL,
        VALID_ENTRIES_NPI_ITEM_COL,
        VALID_ENTRIES_NPI_ACCOUNT_COL,
        VALID_ENTRIES_NPI_CHANNEL_COL,
        VALID_ENTRIES_NPI_REGION_COL,
        VALID_ENTRIES_NPI_PNL_COL,
        VALID_ENTRIES_NPI_DEMAND_DOMAIN_COL,
        VALID_ENTRIES_NPI_LOCATION_COL,
        NPI_DATA_ACCEPTED_FLAG_COL,
        NPI_LAUNCH_DATE_L0_COL,
        NPI_EOL_DATE_L0_COL,
    ]
    # Corrected initiative_stage_fact_file columns (keeping Validationcheck.py format)
    initiative_stage_fact_file_req_output_cols = [
        VERSION_COL,
        INITIATIVE_NAME_OUTPUT_COL,
        Data_validation_COL,
        Stage_association_COL,
        Stage_owner_COL,
        Stage_created_date_COL,
    ]

    # Required columns for NPIInitiativeLevel
    NPI_INITIATIVE_LEVEL_REQUIRED_COLS = [
        o9Constants.INITIATIVE,  # Using existing constant
        o9Constants.DATA_OBJECT,
        o9Constants.NPI_INITIATIVE_LEVEL_ASSOCIATION,
    ]

    # Required columns for NPIAssortments
    NPI_ASSORTMENTS_REQUIRED_COLS = [
        o9Constants.INITIATIVE,  # Using existing constant
        o9Constants.DATA_OBJECT,
        o9Constants.ITEM_NPI_ITEM,
        o9Constants.ACCOUNT_NPI_ACCOUNT,
        o9Constants.CHANNEL_NPI_CHANNEL,
        o9Constants.REGION_NPI_REGION,
        o9Constants.PNL_NPI_PNL,
        o9Constants.DEMAND_DOMAIN_NPI_DEMAND_DOMAIN,
        o9Constants.LOCATION_NPI_LOCATION,
        o9Constants.NPI_ASSOCIATION_L0,
    ]

    # Initiative column constants
    INITIATIVE_COL = o9Constants.INITIATIVE  # Using existing constant
    INITIATIVE_DISPLAY_NAME_COL = o9Constants.INITIATIVE_DISPLAY_NAME

    # Display Name Columns
    DATA_OBJECT_DISPLAYNAME_COL = "Data Object.[Data Object$DisplayName]"
    ITEM_PLANNING_DISPLAYNAME_COL = "Item.[Planning Item$DisplayName]"
    ACCOUNT_PLANNING_DISPLAYNAME_COL = "Account.[Planning Account$DisplayName]"
    CHANNEL_PLANNING_DISPLAYNAME_COL = "Channel.[Planning Channel$DisplayName]"
    REGION_PLANNING_DISPLAYNAME_COL = "Region.[Planning Region$DisplayName]"
    PNL_PLANNING_DISPLAYNAME_COL = "PnL.[Planning PnL$DisplayName]"
    DEMAND_DOMAIN_PLANNING_DISPLAYNAME_COL = "Demand Domain.[Planning Demand Domain$DisplayName]"
    LOCATION_PLANNING_DISPLAYNAME_COL = "Location.[Planning Location$DisplayName]"

    # Required columns for main output (using input column names)
    REQUIRED_OUTPUT_COLS = [
        VERSION_COL,
        EMAIL_COL,
        SEQUENCE_COL,
        INITIATIVE_NAME_INPUT_COL,
        ASSORTMENT_GROUP_COL,
        LEVEL_NAME_COL,
        NPI_ITEM_COL,
        NPI_ACCOUNT_COL,
        NPI_CHANNEL_COL,
        NPI_REGION_COL,
        NPI_PNL_COL,
        NPI_DEMAND_DOMAIN_COL,
        NPI_LOCATION_COL,
        NPI_LAUNCH_DATE_COL,
        NPI_EOL_DATE_COL,
        REJECTED_FLAG_COL,
        VALIDATION_REMARK_COL,
    ]


# ===============================================================================
# COLUMN MAPPING FOR OUTPUT DTYPES
# ===============================================================================

cfg = ValidationConfig

col_mapping = {
    cfg.REJECTED_FLAG_COL: int,
    cfg.VALIDATION_REMARK_COL: str,
    cfg.NPI_INITIATIVE_START_DATE_COL: str,
    cfg.NPI_INITIATIVE_END_DATE_COL: str,
    cfg.LAUNCH_DATE_COL: str,
    cfg.EOL_DATE_COL: str,
    cfg.NPI_INITIATIVE_STATUS_COL: str,
    cfg.NPI_INITIATIVE_CREATED_BY_COL: str,
    cfg.INITIATIVE_OWNER_COL: str,
    cfg.NPI_INITIATIVE_CREATED_DATE_COL: str,
    cfg.NPI_INITIATIVE_LEVEL_ASSOCIATION_COL: int,
    cfg.NPI_ITEM_LEVEL_COL: str,
    cfg.NPI_ACCOUNT_LEVEL_COL: str,
    cfg.NPI_CHANNEL_LEVEL_COL: str,
    cfg.NPI_REGION_LEVEL_COL: str,
    cfg.NPI_PNL_LEVEL_COL: str,
    cfg.NPI_DEMAND_DOMAIN_LEVEL_COL: str,
    cfg.NPI_LOCATION_LEVEL_COL: str,
    cfg.NPI_Inititative_level_status_COL: str,
    cfg.NPI_Initiative_lowest_level_COL: int,
    cfg.NPI_LEVEL_SEQUENCE_COL_L1: int,
    cfg.NPI_fcst_generation_method_l1_COL: str,
    cfg.NPI_Ramp_up_period_l1_COL: "Int64",
    cfg.NPI_Ramp_up_bucket_l1_COL: str,
    cfg.NPI_DATA_ACCEPTED_FLAG_COL: int,
    cfg.NPI_LAUNCH_DATE_L0_COL: str,
    cfg.NPI_EOL_DATE_L0_COL: str,
    cfg.Data_validation_COL: str,
    cfg.Stage_association_COL: str,
    cfg.Stage_owner_COL: str,
    cfg.Stage_created_date_COL: str,
    cfg.INITIATIVE_NAME_OUTPUT_COL: str,
    cfg.INITIATIVE_DISPLAY_NAME_COL: str,
    cfg.INITIATIVE_DESCRIPTION_COL: str,
    cfg.INITIATIVE_TYPE_COL: str,
    cfg.INITIATIVE_TYPE_DISPLAY_NAME_COL: str,
    cfg.INITIATIVE_LEVEL_1_COL: str,
    cfg.INITIATIVE_LEVEL_2_COL: str,
    cfg.INITIATIVE_LEVEL_3_COL: str,
    cfg.INITIATIVE_LEVEL_4_COL: str,
    cfg.INITIATIVE_LEVEL_5_COL: str,
    cfg.INITIATIVE_LEVEL_6_COL: str,
    cfg.INITIATIVE_LEVEL_7_COL: str,
    cfg.INITIATIVE_LEVEL_8_COL: str,
    cfg.INITIATIVE_CATEGORY_COL: str,
    cfg.DATA_OBJECT_COL: str,
    cfg.DATA_OBJECT_DISPLAYNAME_COL: str,
    cfg.PLANNING_ITEM_COL: str,
    cfg.ITEM_PLANNING_DISPLAYNAME_COL: str,
    cfg.PLANNING_ACCOUNT_COL: str,
    cfg.ACCOUNT_PLANNING_DISPLAYNAME_COL: str,
    cfg.PLANNING_CHANNEL_COL: str,
    cfg.CHANNEL_PLANNING_DISPLAYNAME_COL: str,
    cfg.PLANNING_REGION_COL: str,
    cfg.REGION_PLANNING_DISPLAYNAME_COL: str,
    cfg.PLANNING_PNL_COL: str,
    cfg.PNL_PLANNING_DISPLAYNAME_COL: str,
    cfg.PLANNING_DEMAND_DOMAIN_COL: str,
    cfg.DEMAND_DOMAIN_PLANNING_DISPLAYNAME_COL: str,
    cfg.PLANNING_LOCATION_COL: str,
    cfg.LOCATION_PLANNING_DISPLAYNAME_COL: str,
    cfg.VALID_ENTRIES_NPI_ITEM_COL: str,
    cfg.VALID_ENTRIES_NPI_ACCOUNT_COL: str,
    cfg.VALID_ENTRIES_NPI_CHANNEL_COL: str,
    cfg.VALID_ENTRIES_NPI_REGION_COL: str,
    cfg.VALID_ENTRIES_NPI_PNL_COL: str,
    cfg.VALID_ENTRIES_NPI_DEMAND_DOMAIN_COL: str,
    cfg.VALID_ENTRIES_NPI_LOCATION_COL: str,
    cfg.VERSION_COL: str,
    cfg.EMAIL_COL: str,
    cfg.SEQUENCE_COL: str,
    cfg.INITIATIVE_NAME_INPUT_COL: str,
    cfg.INITIATIVE_DESC_COL: str,
    cfg.LEVEL_NAME_COL: str,
    cfg.ASSORTMENT_GROUP_COL: str,
    cfg.NPI_ITEM_COL: str,
    cfg.NPI_ACCOUNT_COL: str,
    cfg.NPI_CHANNEL_COL: str,
    cfg.NPI_REGION_COL: str,
    cfg.NPI_PNL_COL: str,
    cfg.NPI_DEMAND_DOMAIN_COL: str,
    cfg.NPI_LOCATION_COL: str,
    cfg.NPI_LAUNCH_DATE_COL: str,
    cfg.NPI_EOL_DATE_COL: str,
}

# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================


def fetch_npi_initiatives(num_initiatives=50):
    """Fetch unique NPI initiative IDs using web API."""
    logger.info("Fetching {} NPI initiative IDs using web API".format(num_initiatives))

    # Initialize O9API client like in member_creation.py
    tenant_api_client = O9API(verify=False, tenant_url="")
    query = "select NextLabel([NPIInitiativeID]);"

    labels_set = set()
    execution_count = 0
    start_time = time.time()

    # Initialize session
    if tenant_api_client.initialize_session():
        logger.info("Session initialized successfully for NPI initiative fetching")

        while len(labels_set) < num_initiatives:
            result = tenant_api_client.run_ibpl_query(query)
            execution_count += 1

            # Extract the actual value from the result
            if isinstance(result, dict):
                # The result is a dictionary, extract the actual value
                # Usually the first (and only) value in the result
                actual_value = list(result.values())[0] if result else None
            else:
                actual_value = result

            if actual_value and actual_value not in labels_set:
                labels_set.add(actual_value)

    elif tenant_api_client.set_api_key():
        logger.warning("Retry with API KEY successful for NPI initiative fetching")

        while len(labels_set) < num_initiatives:
            result = tenant_api_client.run_ibpl_query(query)
            execution_count += 1

            # Extract the actual value from the result
            if isinstance(result, dict):
                # The result is a dictionary, extract the actual value
                actual_value = list(result.values())[0] if result else None
            else:
                actual_value = result

            if actual_value and actual_value not in labels_set:
                labels_set.add(actual_value)
    else:
        logger.error("Authentication unsuccessful for NPI initiative fetching")
        raise Exception("Failed to authenticate for NPI initiative ID generation")

    logger.info(
        "Query executed {} times in {:.2f} seconds".format(
            execution_count, time.time() - start_time
        )
    )
    logger.info("Final set of NPI initiative IDs collected: {}".format(sorted(labels_set)))
    logger.info("Total unique initiative IDs generated: {}".format(len(labels_set)))
    return pd.DataFrame(sorted(labels_set), columns=["Initiative"])


def _prepare_master_data_config(*master_dataframes):
    """Prepare master data configuration for validation."""
    cfg = ValidationConfig
    entity_names = [
        cfg.Item_name,
        cfg.Account_name,
        cfg.Channel_name,
        cfg.Region_name,
        cfg.PnL_name,
        cfg.DemandDomain_name,
        cfg.Location_name,
    ]
    return {entity: {"dataframe": df} for entity, df in zip(entity_names, master_dataframes)}


def _initialize_empty_outputs():
    """Initialize empty output dataframes with proper column structure."""
    cfg = ValidationConfig

    empty_initiative_dim_data = pd.DataFrame(columns=cfg.initiative_dim_data_req_output_cols)
    empty_initiative_fact_data = pd.DataFrame(columns=cfg.initiative_fact_data_req_output_cols)
    empty_initiative_level_fact_data = pd.DataFrame(
        columns=cfg.initiative_level_fact_data_req_output_cols
    )
    empty_valid_entries = pd.DataFrame(columns=cfg.valid_entries_output_req_cols)
    empty_initiative_stage_fact_file = pd.DataFrame(
        columns=cfg.initiative_stage_fact_file_req_output_cols
    )

    return (
        empty_initiative_dim_data,
        empty_initiative_fact_data,
        empty_initiative_level_fact_data,
        empty_valid_entries,
        empty_initiative_stage_fact_file,
    )


def _prepare_remarktable(file_upload_df):
    """Prepare and initialize the main validation table."""
    cfg = ValidationConfig

    if file_upload_df.empty:
        logger.info("No file upload data found")
        return pd.DataFrame(columns=cfg.REQUIRED_OUTPUT_COLS)

    # Debug: Log original email and sequence values
    logger.info(f"Original FileUpload shape: {file_upload_df.shape}")
    logger.info(f"Original FileUpload columns: {file_upload_df.columns.tolist()}")

    # NEW: Normalize column names to match expected format
    logger.info("Normalizing column names to match expected format...")

    # Create a copy to avoid modifying the original
    file_upload_df = file_upload_df.copy()

    # Define column name mappings
    column_mappings = {
        "Personnel.[Email]": "[Personnel].[Email]",
        "Sequence.[Sequence]": "[Sequence].[Sequence]",
    }

    # Apply column name mappings
    for old_name, new_name in column_mappings.items():
        if old_name in file_upload_df.columns:
            logger.info(f"Renaming column '{old_name}' to '{new_name}'")
            file_upload_df = file_upload_df.rename(columns={old_name: new_name})
        else:
            logger.info(f"Column '{old_name}' not found in DataFrame")

    logger.info(f"After normalization - FileUpload columns: {file_upload_df.columns.tolist()}")

    # Enhanced debugging for column access issues
    logger.info(f"Looking for EMAIL_COL: '{cfg.EMAIL_COL}'")
    logger.info(f"Looking for SEQUENCE_COL: '{cfg.SEQUENCE_COL}'")

    # Check exact column matches
    email_col_found = cfg.EMAIL_COL in file_upload_df.columns
    sequence_col_found = cfg.SEQUENCE_COL in file_upload_df.columns

    logger.info(f"EMAIL_COL found: {email_col_found}")
    logger.info(f"SEQUENCE_COL found: {sequence_col_found}")

    # If not found, check for similar column names
    if not email_col_found:
        similar_email_cols = [
            col
            for col in file_upload_df.columns
            if "email" in col.lower() or "personnel" in col.lower()
        ]
        logger.info(f"Similar email columns found: {similar_email_cols}")

    if not sequence_col_found:
        similar_sequence_cols = [col for col in file_upload_df.columns if "sequence" in col.lower()]
        logger.info(f"Similar sequence columns found: {similar_sequence_cols}")

    if cfg.EMAIL_COL in file_upload_df.columns:
        logger.info(
            f"Original {cfg.EMAIL_COL} values: {file_upload_df[cfg.EMAIL_COL].head().tolist()}"
        )
        logger.info(f"Original {cfg.EMAIL_COL} dtype: {file_upload_df[cfg.EMAIL_COL].dtype}")
    else:
        logger.warning(f"Column {cfg.EMAIL_COL} not found in FileUpload")

    if cfg.SEQUENCE_COL in file_upload_df.columns:
        logger.info(
            f"Original {cfg.SEQUENCE_COL} values: {file_upload_df[cfg.SEQUENCE_COL].head().tolist()}"
        )
        logger.info(f"Original {cfg.SEQUENCE_COL} dtype: {file_upload_df[cfg.SEQUENCE_COL].dtype}")
    else:
        logger.warning(f"Column {cfg.SEQUENCE_COL} not found in FileUpload")

    # Create validation table
    invalid_entries = file_upload_df.copy()

    # Initialize validation columns
    invalid_entries[cfg.REJECTED_FLAG_COL] = False
    invalid_entries[cfg.VALIDATION_REMARK_COL] = ""

    # Ensure all required columns exist
    for col in cfg.REQUIRED_OUTPUT_COLS:
        if col not in invalid_entries.columns:
            logger.warning(f"Adding missing column: {col}")
            invalid_entries[col] = ""

    # Add initiative description column if not present
    if cfg.INITIATIVE_DESC_COL not in invalid_entries.columns:
        invalid_entries[cfg.INITIATIVE_DESC_COL] = ""

    # Debug: Log final email and sequence values
    logger.info(f"Final invalid_entries shape: {invalid_entries.shape}")
    if cfg.EMAIL_COL in invalid_entries.columns:
        logger.info(
            f"Final {cfg.EMAIL_COL} values: {invalid_entries[cfg.EMAIL_COL].head().tolist()}"
        )
    if cfg.SEQUENCE_COL in invalid_entries.columns:
        logger.info(
            f"Final {cfg.SEQUENCE_COL} values: {invalid_entries[cfg.SEQUENCE_COL].head().tolist()}"
        )

    logger.info(f"Prepared validation table with {len(invalid_entries)} records")
    return invalid_entries


def _run_all_validations(
    invalid_entries,
    master_dfs_config,
    global_npi_levels,
    npi_level,
    npi_initiative,
    npi_initiative_level,
    npi_initiative_status,
    npi_assortments,
):
    """Run all validation checks in the specified sequence."""
    cfg = ValidationConfig

    logger.info("Starting validation sequence according to specified order")

    # 1. Assortment Group Number empty check - Rejects if empty
    logger.info("1. Running Assortment Group Number empty check")
    empty_assortment_mask = (
        invalid_entries[cfg.ASSORTMENT_GROUP_COL].isna()
        | (invalid_entries[cfg.ASSORTMENT_GROUP_COL] == "")
        | (invalid_entries[cfg.ASSORTMENT_GROUP_COL].astype(str).str.strip() == "")
    )
    if empty_assortment_mask.sum() > 0:
        invalid_entries.loc[empty_assortment_mask, cfg.REJECTED_FLAG_COL] = True
        invalid_entries.loc[empty_assortment_mask, cfg.VALIDATION_REMARK_COL] = (
            "Assortment Group Number is required"
        )

    # 2. NPI Level Name empty check
    logger.info("2. Running NPI Level Name empty check")
    empty_level_mask = (
        invalid_entries[cfg.LEVEL_NAME_COL].isna()
        | (invalid_entries[cfg.LEVEL_NAME_COL] == "")
        | (invalid_entries[cfg.LEVEL_NAME_COL].astype(str).str.strip() == "")
    )
    if empty_level_mask.sum() > 0:
        invalid_entries.loc[empty_level_mask, cfg.REJECTED_FLAG_COL] = True
        append_validation_remark_vectorized(
            invalid_entries, empty_level_mask, cfg.VALIDATION_REMARK_COL, "Level name is required"
        )

    # 3. NPI Level Name validity check - Validates against NPILevel dataframe (including display names)
    logger.info("3. Running NPI Level Name validity check")
    invalid_entries = run_level_name_validation(
        invalid_entries,
        cfg.LEVEL_NAME_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
        npi_level,
    )

    # 4. Entity empty checks - Validates Item, Account, Channel, Region, PnL, Demand Domain, Location names are not empty
    logger.info("4. Running Entity empty checks")
    invalid_entries = run_entity_empty_checks(
        invalid_entries, cfg.REJECTED_FLAG_COL, cfg.VALIDATION_REMARK_COL
    )

    # 5. NPI Product Launch Date empty check - Rejects if empty
    logger.info("5. Running NPI Product Launch Date empty check")
    empty_launch_date_mask = (
        invalid_entries[cfg.NPI_LAUNCH_DATE_COL].isna()
        | (invalid_entries[cfg.NPI_LAUNCH_DATE_COL] == "")
        | (invalid_entries[cfg.NPI_LAUNCH_DATE_COL].astype(str).str.strip() == "")
        | (invalid_entries[cfg.NPI_LAUNCH_DATE_COL].astype(str).str.strip() == "nan")
    )
    if empty_launch_date_mask.sum() > 0:
        append_validation_remark_vectorized(
            invalid_entries,
            empty_launch_date_mask,
            cfg.VALIDATION_REMARK_COL,
            "Product Launch Date is required",
        )
        invalid_entries.loc[empty_launch_date_mask, cfg.REJECTED_FLAG_COL] = True

    # 6. Date validation for Launch Date and EOL Date - Validates date formats
    logger.info("6. Running Date validation for Launch Date and EOL Date")
    invalid_entries = run_date_validation(
        invalid_entries,
        cfg.NPI_LAUNCH_DATE_COL,
        cfg.NPI_EOL_DATE_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
    )

    # 7. DisplayName Mapping (apply_optimized_displayname_mapping)
    logger.info("7. Running DisplayName Mapping")
    invalid_entries = apply_optimized_displayname_mapping(invalid_entries, master_dfs_config)

    # 8. Initiative name consistency check - Ensures same assortment group has same initiative name
    logger.info("8. Running Initiative name consistency check")
    invalid_entries = run_initiative_consistency_check(
        invalid_entries,
        cfg.ASSORTMENT_GROUP_COL,
        cfg.INITIATIVE_NAME_INPUT_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
    )

    # 9. Initiative name validity check - Validates against NPIInitiative dataframe (including display names)
    logger.info("9. Running Initiative name validity check")
    invalid_entries = run_initiative_validity_check(
        invalid_entries,
        cfg.INITIATIVE_NAME_INPUT_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
        npi_initiative,
    )

    # 10. NPI Initiative Level Association Validation (check_npi_initiative_level_association)
    logger.info("10. Running NPI Initiative Level Association Validation")
    invalid_entries = check_npi_initiative_level_association(
        invalid_entries,
        cfg.INITIATIVE_NAME_INPUT_COL,
        cfg.LEVEL_NAME_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
        npi_initiative_level,
    )

    # 11. Initiative status check - Validates initiative status from NPIInitiativeStatus
    logger.info("11. Running Initiative status check")
    invalid_entries = run_initiative_status_check(
        invalid_entries,
        cfg.INITIATIVE_NAME_INPUT_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
        npi_initiative_status,
    )

    # 12. Initiative level status check - Validates initiative level status from NPIInitiativeLevel
    logger.info("12. Running Initiative level status check")
    invalid_entries = run_initiative_level_status_check(
        invalid_entries,
        cfg.INITIATIVE_NAME_INPUT_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
        npi_initiative_level,
    )

    # 13. Initiative description consistency check - When initiative names are blank, validates description consistency
    logger.info("13. Running Initiative description consistency check")
    invalid_entries = run_initiative_description_consistency_check(
        invalid_entries,
        cfg.ASSORTMENT_GROUP_COL,
        cfg.INITIATIVE_NAME_INPUT_COL,
        cfg.INITIATIVE_DESC_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
    )

    # 14. MANDATORY: Entity validation (always runs regardless of rejection status) - MOVED BEFORE NPIAssortment validation
    logger.info("14. Running MANDATORY Entity validation (always runs)")
    invalid_entries = run_mandatory_entity_validation(
        invalid_entries,
        cfg.LEVEL_NAME_COL,
        cfg.NPI_ITEM_COL,
        cfg.NPI_ACCOUNT_COL,
        cfg.NPI_CHANNEL_COL,
        cfg.NPI_REGION_COL,
        cfg.NPI_PNL_COL,
        cfg.NPI_DEMAND_DOMAIN_COL,
        cfg.NPI_LOCATION_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
        global_npi_levels,
        master_dfs_config,
    )

    # 15. NPIAssortment Association Validation (check_npi_assortment_association)
    logger.info("15. Running NPIAssortment Association Validation")
    invalid_entries = check_npi_assortment_association(
        invalid_entries,
        cfg.ASSORTMENT_GROUP_COL,
        cfg.INITIATIVE_NAME_INPUT_COL,
        cfg.LEVEL_NAME_COL,
        cfg.NPI_ITEM_COL,
        cfg.NPI_ACCOUNT_COL,
        cfg.NPI_CHANNEL_COL,
        cfg.NPI_REGION_COL,
        cfg.NPI_PNL_COL,
        cfg.NPI_DEMAND_DOMAIN_COL,
        cfg.NPI_LOCATION_COL,
        cfg.REJECTED_FLAG_COL,
        cfg.VALIDATION_REMARK_COL,
        npi_assortments,
    )

    # 16. Group Rejection Propagation (propagate_group_rejections)
    logger.info("16. Running Group Rejection Propagation")
    invalid_entries = propagate_group_rejections(
        invalid_entries, cfg.ASSORTMENT_GROUP_COL, cfg.REJECTED_FLAG_COL
    )

    logger.info("Completed all validation checks in specified sequence")
    return invalid_entries


def _create_all_outputs(
    invalid_entries,
    empty_initiative_dim_data,
    empty_initiative_fact_data,
    empty_initiative_level_fact_data,
    empty_valid_entries,
    empty_initiative_stage_fact_file,
    global_npi_levels,
    data_validation,
):
    """Create all output tables from validated data."""
    cfg = ValidationConfig

    # Filter correct data and create initiatives
    correct_df = invalid_entries[~invalid_entries[cfg.REJECTED_FLAG_COL]].copy()
    logger.info(f"Found {len(correct_df)} correct records for initiative creation")

    if correct_df.empty:
        logger.info("No correct records found - returning empty outputs")
        return (
            empty_initiative_dim_data,
            empty_initiative_fact_data,
            empty_initiative_level_fact_data,
            empty_valid_entries,
            empty_initiative_stage_fact_file,
        )

    # STEP 1: First propagate existing initiative names within assortment groups
    logger.info("Step 1: Propagating existing initiative names within assortment groups")
    correct_df = propagate_existing_initiative_names(
        correct_df, cfg.ASSORTMENT_GROUP_COL, cfg.INITIATIVE_NAME_INPUT_COL
    )

    # STEP 2: Create initiatives only for records that still don't have initiative names
    logger.info("Step 2: Creating initiatives for records that still don't have initiative names")
    (
        updated_correct_df,
        initiative_dim_data,
        initiative_fact_data,
        initiative_level_fact_data,
        valid_entries,
        initiative_stage_fact_file,
    ) = create_initiatives_for_missing(
        correct_df,
        cfg.ASSORTMENT_GROUP_COL,
        cfg.INITIATIVE_NAME_INPUT_COL,
        cfg.INITIATIVE_DESC_COL,
        cfg.EMAIL_COL,
        cfg.NPI_LAUNCH_DATE_COL,
        cfg.NPI_EOL_DATE_COL,
        global_npi_levels,
        data_validation,
    )

    # NEW: STEP 3: Add existing intersections from invalid_entries where NPI Data Rejected Flag = False
    logger.info(
        "Step 3: Adding existing intersections from invalid_entries where NPI Data Rejected Flag = False"
    )
    existing_valid_entries = create_existing_valid_entries(invalid_entries, cfg)

    # Combine newly created valid_entries with existing valid entries
    if not existing_valid_entries.empty:
        logger.info(
            f"Adding {len(existing_valid_entries)} existing valid entries to valid_entries output"
        )
        valid_entries = pd.concat([valid_entries, existing_valid_entries], ignore_index=True)
        logger.info(f"Combined valid_entries now has {len(valid_entries)} total records")

    # Update main validation table with newly created initiatives if needed
    if not updated_correct_df.empty:
        for idx in updated_correct_df.index:
            if idx in invalid_entries.index:
                invalid_entries.loc[idx, cfg.INITIATIVE_NAME_INPUT_COL] = updated_correct_df.loc[
                    idx, cfg.INITIATIVE_NAME_INPUT_COL
                ]

    return (
        initiative_dim_data,
        initiative_fact_data,
        initiative_level_fact_data,
        valid_entries,
        initiative_stage_fact_file,
    )


def _finalize_remarktable_output(invalid_entries):
    """Clean up and format the final validation output."""
    cfg = ValidationConfig

    # Debug: Log DataFrame info before processing
    logger.info(f"Before finalization: DataFrame shape = {invalid_entries.shape}")
    logger.info(f"Before finalization: Available columns = {invalid_entries.columns.tolist()}")
    logger.info(f"Before finalization: Required columns = {cfg.invalid_entries_output_req_cols}")

    # Debug: Log email and sequence values before processing
    if cfg.EMAIL_COL in invalid_entries.columns:
        logger.info(
            f"Before finalization: {cfg.EMAIL_COL} values: {invalid_entries[cfg.EMAIL_COL].head().tolist()}"
        )
    if cfg.SEQUENCE_COL in invalid_entries.columns:
        logger.info(
            f"Before finalization: {cfg.SEQUENCE_COL} values: {invalid_entries[cfg.SEQUENCE_COL].head().tolist()}"
        )

    # Remove unnecessary columns
    if cfg.INITIATIVE_DESC_COL in invalid_entries.columns:
        invalid_entries.drop(columns=[cfg.INITIATIVE_DESC_COL], inplace=True)

    # Remove temporary "Original" columns
    original_cols_to_remove = [col for col in invalid_entries.columns if col.endswith(" Original")]
    if original_cols_to_remove:
        invalid_entries.drop(columns=original_cols_to_remove, inplace=True)
        logger.info(f"Removed temporary columns: {len(original_cols_to_remove)}")

    # Check if all required columns exist
    missing_cols = [
        col for col in cfg.invalid_entries_output_req_cols if col not in invalid_entries.columns
    ]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {invalid_entries.columns.tolist()}")

        # Create missing columns with empty values
        for col in missing_cols:
            invalid_entries[col] = ""
            logger.info(f"Added missing column '{col}' with empty values")

    # Reorder columns to match output requirements (5-column validation summary)
    invalid_entries = invalid_entries.reindex(
        columns=cfg.invalid_entries_output_req_cols, fill_value=""
    )

    # Debug: Log email and sequence values after processing
    logger.info(f"After finalization: DataFrame shape = {invalid_entries.shape}")
    if cfg.EMAIL_COL in invalid_entries.columns:
        logger.info(
            f"After finalization: {cfg.EMAIL_COL} values: {invalid_entries[cfg.EMAIL_COL].head().tolist()}"
        )
    if cfg.SEQUENCE_COL in invalid_entries.columns:
        logger.info(
            f"After finalization: {cfg.SEQUENCE_COL} values: {invalid_entries[cfg.SEQUENCE_COL].head().tolist()}"
        )

    return invalid_entries


def append_validation_remark_vectorized(invalid_entries, mask, validation_remark_col, new_remark):
    """Efficient vectorized append of validation remarks with duplicate prevention."""
    # Convert mask to boolean if it isn't already (safety check)
    if hasattr(mask, "dtype") and mask.dtype != "bool":
        mask = mask.astype(bool)

    if mask.sum() > 0:
        # Ensure validation_remark_col is string type
        if validation_remark_col not in invalid_entries.columns:
            invalid_entries[validation_remark_col] = ""
        invalid_entries[validation_remark_col] = (
            invalid_entries[validation_remark_col].fillna("").astype(str)
        )

        # Check for existing remarks that don't already contain this message
        existing_remarks_mask = mask & (invalid_entries[validation_remark_col] != "")
        new_remarks_mask = mask & (invalid_entries[validation_remark_col] == "")

        # Filter out records that already have this exact message
        if existing_remarks_mask.sum() > 0:
            # Check which existing remarks don't already contain this message
            existing_remarks = invalid_entries.loc[existing_remarks_mask, validation_remark_col]
            not_already_present = ~existing_remarks.str.contains(new_remark, case=False, na=False)
            existing_remarks_mask = existing_remarks_mask & not_already_present

        # Vectorized updates
        if existing_remarks_mask.sum() > 0:
            invalid_entries.loc[existing_remarks_mask, validation_remark_col] = (
                invalid_entries.loc[existing_remarks_mask, validation_remark_col]
                + "; "
                + new_remark
            )

        if new_remarks_mask.sum() > 0:
            invalid_entries.loc[new_remarks_mask, validation_remark_col] = new_remark


def create_entity_level_mappings(global_npi_levels_df):
    """Create mappings for NPI Level to Entity Level configuration."""
    level_mappings = {}
    cfg = ValidationConfig

    if not global_npi_levels_df.empty:
        level_col = cfg.DATA_OBJECT_COL
        entity_level_cols = {
            cfg.Item_name: cfg.global_npi_item_level,
            cfg.Account_name: cfg.global_npi_account_level,
            cfg.Channel_name: cfg.global_npi_channel_level,
            cfg.Region_name: cfg.global_npi_region_level,
            cfg.PnL_name: cfg.global_npi_pnl_level,
            cfg.DemandDomain_name: cfg.global_npi_demand_domain_level,
            cfg.Location_name: cfg.global_npi_location_level,
        }

        if level_col in global_npi_levels_df.columns:
            for _, row in global_npi_levels_df.iterrows():
                npi_level = str(row[level_col]).strip()
                entity_levels = {}

                for entity, col in entity_level_cols.items():
                    if col in global_npi_levels_df.columns:
                        entity_levels[entity] = str(row[col]).strip() if pd.notna(row[col]) else ""

                level_mappings[npi_level] = entity_levels

    return level_mappings


def create_master_data_lookups(master_dfs_config):
    """Create optimized lookup dictionaries for master data validation."""
    master_lookups = {}

    for entity_name, config in master_dfs_config.items():
        df = config["dataframe"]

        entity_lookups = {}

        if not df.empty:
            # Dynamically detect all available columns in the master dataframe
            available_columns = df.columns.tolist()

            # Create lookup sets for each column that contains valid data
            for column_name in available_columns:
                if column_name in df.columns:
                    # Create set of valid values for this column
                    valid_values = set(df[column_name].dropna().astype(str).str.strip())
                    # Remove empty strings
                    valid_values = {v for v in valid_values if v != ""}

                    if valid_values:  # Only add if there are valid values
                        entity_lookups[column_name] = valid_values
                        logger.debug(
                            f"Created lookup for {entity_name}.{column_name} with {len(valid_values)} values"
                        )

        master_lookups[entity_name] = entity_lookups
        logger.info(f"Created {len(entity_lookups)} column lookups for {entity_name}")

    return master_lookups


# mapping the displayname to the original value
# iteration over the master dataframes and create the mappings, vlaue wise
def create_displayname_to_original_mappings(master_dfs_config):
    """Create mappings from DisplayName values to original values using master dataframe."""
    displayname_mappings = {}

    for entity_name, config in master_dfs_config.items():
        df = config["dataframe"]
        entity_mappings = {}

        if not df.empty:
            # Find DisplayName and original columns
            displayname_cols = [col for col in df.columns if "$DisplayName" in col]

            for displayname_col in displayname_cols:
                # Find corresponding original column (remove $DisplayName suffix)
                original_col = displayname_col.replace("$DisplayName", "")

                if original_col in df.columns:
                    # Create mapping dictionary from DisplayName to original value
                    mapping_dict = {}

                    for _, row in df.iterrows():
                        display_val = (
                            str(row[displayname_col]).strip()
                            if pd.notna(row[displayname_col])
                            else ""
                        )
                        original_val = (
                            str(row[original_col]).strip() if pd.notna(row[original_col]) else ""
                        )

                        if (
                            display_val
                            and original_val
                            and display_val != "nan"
                            and original_val != "nan"
                        ):
                            mapping_dict[display_val] = original_val

                    if mapping_dict:
                        entity_mappings[displayname_col] = {
                            "original_column": original_col,
                            "mapping": mapping_dict,
                        }
                        logger.debug(
                            f"Created DisplayName mapping for {entity_name}.{displayname_col} -> {original_col} with {len(mapping_dict)} mappings"
                        )

        displayname_mappings[entity_name] = entity_mappings
        logger.info(f"Created {len(entity_mappings)} DisplayName mappings for {entity_name}")

    return displayname_mappings


# this function checks only the empty necessary columns


def apply_optimized_displayname_mapping(invalid_entries, master_dfs_config):
    """Optimized DisplayName mapping using vectorized operations with sequential mapping."""
    logger.info("Applying DisplayName mapping")

    # Pre-compute all mappings
    displayname_mappings = create_displayname_to_original_mappings(master_dfs_config)
    cfg = ValidationConfig

    # Apply mappings using vectorized operations
    mapping_count = 0
    for invalid_entries_col, entity_key in cfg.ENTITY_MAPPINGS.items():
        if invalid_entries_col in invalid_entries.columns and entity_key in displayname_mappings:

            original_col_name = f"{invalid_entries_col} Original"
            invalid_entries[original_col_name] = invalid_entries[invalid_entries_col].copy()

            entity_mappings = displayname_mappings[entity_key]

            # Get values that need mapping
            mask = invalid_entries[original_col_name].notna() & (
                invalid_entries[original_col_name].astype(str).str.strip() != ""
            )
            if mask.sum() > 0:
                values_to_map = invalid_entries.loc[mask, original_col_name].astype(str).str.strip()

                # Try each DisplayName mapping sequentially (instead of combining them)
                mapped_values = values_to_map.copy()  # Start with original values
                mapping_used = {}  # Track which mapping was used for each value

                for displayname_col, mapping_info in entity_mappings.items():
                    current_mapping = mapping_info["mapping"]
                    if current_mapping:
                        # Apply this specific mapping
                        current_mapped = values_to_map.map(current_mapping)

                        # Update only values that were successfully mapped and not already mapped
                        successful_mask = current_mapped.notna() & mapped_values.eq(values_to_map)
                        if successful_mask.sum() > 0:
                            mapped_values.loc[successful_mask] = current_mapped.loc[successful_mask]
                            # Track which mapping was used
                            for idx in successful_mask[successful_mask].index:
                                mapping_used[values_to_map.loc[idx]] = displayname_col

                            logger.debug(
                                f"Mapped {successful_mask.sum()} values using {displayname_col}"
                            )

                # Update the dataframe with successfully mapped values
                invalid_entries.loc[mask, original_col_name] = mapped_values
                total_mapped = (mapped_values != values_to_map).sum()
                mapping_count += total_mapped

                # Log mapping statistics
                if total_mapped > 0:
                    logger.info(
                        f"Successfully mapped {total_mapped} values for {invalid_entries_col}"
                    )
                    for display_col, count in (
                        pd.Series(list(mapping_used.values())).value_counts().items()
                    ):
                        logger.debug(f"  {count} values mapped using {display_col}")
                else:
                    logger.debug(f"No DisplayName mappings applied for {invalid_entries_col}")

    logger.info(f"Completed DisplayName mapping: {mapping_count} values mapped")
    return invalid_entries


def validate_entities_vectorized(
    invalid_entries,
    level_mask,
    col_name,
    entity_key,
    required_level,
    master_lookups,
    rejected_flag_col,
    validation_remark_col,
):
    """Vectorized entity validation against master data."""
    validation_count = 0
    cfg = ValidationConfig

    if entity_key in master_lookups:
        entity_master_lookup = master_lookups[entity_key]

        # Get entity values for this level
        entity_values = invalid_entries.loc[level_mask, col_name].astype(str).str.strip()
        non_empty_values = entity_values[entity_values != ""]

        if len(non_empty_values) > 0:
            # Build possible column names to check
            entity_prefix = entity_key.replace(cfg.DemandDomain_name, "Demand Domain")
            possible_columns = [
                f"{entity_prefix}.[{required_level}]",
                f"{entity_prefix}.[{required_level}$DisplayName]",
                required_level,
            ]

            # Find valid values across all possible columns
            valid_values = set()
            for col in possible_columns:
                if col in entity_master_lookup:
                    valid_values.update(entity_master_lookup[col])

            # Vectorized validation
            if valid_values:
                invalid_mask = (
                    level_mask
                    & (invalid_entries[col_name].astype(str).str.strip() != "")
                    & (~invalid_entries[col_name].astype(str).str.strip().isin(valid_values))
                )
                if invalid_mask.sum() > 0:
                    entity_short_name = entity_key.replace(cfg.DemandDomain_name, "Demand Domain")
                    remark = f"{entity_short_name} name doesn't exists"
                    append_validation_remark_vectorized(
                        invalid_entries, invalid_mask, validation_remark_col, remark
                    )
                    invalid_entries.loc[invalid_mask, rejected_flag_col] = True
                    validation_count = invalid_mask.sum()

    return validation_count


def propagate_existing_initiative_names(invalid_entries, assortment_grp_col, initiative_name_col):
    """Propagate existing initiative names within the same (assortment group, email) to records without initiative names."""
    logger.info(
        "Propagating existing initiative names within assortment group and email intersections"
    )
    cfg = ValidationConfig
    grouped = invalid_entries.groupby([assortment_grp_col, cfg.EMAIL_COL])
    propagation_count = 0
    for (assortment_group, email), group_df in grouped:
        if (
            pd.isna(assortment_group)
            or str(assortment_group).strip() == ""
            or pd.isna(email)
            or str(email).strip() == ""
        ):
            continue
        existing_initiatives = group_df[initiative_name_col].dropna()
        existing_initiatives = existing_initiatives[
            existing_initiatives.astype(str).str.strip() != ""
        ]
        if len(existing_initiatives) > 0:
            existing_initiative = existing_initiatives.iloc[0]
            empty_initiative_mask = (
                (group_df[initiative_name_col].isna())
                | (group_df[initiative_name_col] == "")
                | (group_df[initiative_name_col].astype(str).str.strip() == "")
            )
            if empty_initiative_mask.sum() > 0:
                group_indices = group_df[empty_initiative_mask].index
                invalid_entries.loc[group_indices, initiative_name_col] = existing_initiative
                propagation_count += empty_initiative_mask.sum()
                logger.info(
                    f"Assortment group '{assortment_group}', email '{email}': Propagated initiative '{existing_initiative}' to {empty_initiative_mask.sum()} records"
                )
    logger.info(f"Completed initiative name propagation: {propagation_count} records updated")
    return invalid_entries


def create_initiatives_for_missing(
    correct_df,
    assortment_grp_col,
    initiative_name_col,
    initiative_desc_col,
    email_col,
    launch_date_col,
    eol_date_col,
    global_npi_levels_df,
    data_validation_df,
):
    """Create initiatives for records that don't have initiative names. Now per (assortment group, email) intersection."""
    logger.info("Starting initiative creation for missing initiative names")
    cfg = ValidationConfig
    missing_initiative_mask = (
        correct_df[initiative_name_col].isna()
        | (correct_df[initiative_name_col] == "")
        | (correct_df[initiative_name_col].astype(str).str.strip() == "")
    ) & (
        correct_df[assortment_grp_col].notna()
        & (correct_df[assortment_grp_col] != "")
        & (correct_df[assortment_grp_col].astype(str).str.strip() != "")
        & correct_df[email_col].notna()
        & (correct_df[email_col] != "")
        & (correct_df[email_col].astype(str).str.strip() != "")
    )
    missing_initiative_df = correct_df[missing_initiative_mask].copy()
    if missing_initiative_df.empty:
        logger.info(
            "No records found without initiative names and with valid assortment group numbers and emails"
        )
        return (
            correct_df,
            pd.DataFrame(columns=cfg.initiative_dim_data_req_output_cols),
            pd.DataFrame(columns=cfg.initiative_fact_data_req_output_cols),
            pd.DataFrame(columns=cfg.initiative_level_fact_data_req_output_cols),
            pd.DataFrame(columns=cfg.valid_entries_output_req_cols),
            pd.DataFrame(columns=cfg.initiative_stage_fact_file_req_output_cols),
        )
    logger.info(
        f"Found {len(missing_initiative_df)} records without initiative names but with valid assortment group numbers and emails"
    )
    unique_assortment_email = missing_initiative_df[
        [assortment_grp_col, email_col]
    ].drop_duplicates()
    existing_initiative_mask = (
        correct_df[initiative_name_col].notna()
        & (correct_df[initiative_name_col] != "")
        & (correct_df[initiative_name_col].astype(str).str.strip() != "")
        & correct_df[assortment_grp_col].notna()
        & (correct_df[assortment_grp_col] != "")
        & (correct_df[assortment_grp_col].astype(str).str.strip() != "")
        & correct_df[email_col].notna()
        & (correct_df[email_col] != "")
        & (correct_df[email_col].astype(str).str.strip() != "")
    )
    existing_assortment_initiatives = (
        correct_df[existing_initiative_mask]
        .groupby([assortment_grp_col, email_col])[initiative_name_col]
        .first()
    )
    assortment_to_initiative = {}
    groups_needing_new_initiatives = []
    for _, row in unique_assortment_email.iterrows():
        key = (row[assortment_grp_col], row[email_col])
        if key in existing_assortment_initiatives.index:
            existing_initiative = existing_assortment_initiatives[key]
            assortment_to_initiative[key] = existing_initiative
            logger.info(
                f"Assortment group {key[0]}, email {key[1]} will use existing initiative: {existing_initiative}"
            )
        else:
            groups_needing_new_initiatives.append(key)
    if groups_needing_new_initiatives:
        num_initiatives_needed = len(groups_needing_new_initiatives)
        logger.info(
            f"Need to create {num_initiatives_needed} new initiatives for assortment group/email pairs: {groups_needing_new_initiatives}"
        )
        initiative_ids_df = fetch_npi_initiatives(num_initiatives_needed)
        for i, key in enumerate(groups_needing_new_initiatives):
            initiative_id = initiative_ids_df.iloc[i]["Initiative"]
            initiative_name = str(initiative_id)
            assortment_to_initiative[key] = initiative_name
            logger.info(
                f"Created new initiative {initiative_name} for assortment group {key[0]}, email {key[1]}"
            )
    else:
        logger.info(
            "No new initiatives needed - all assortment group/email pairs already have existing initiatives"
        )
    updated_correct_df = correct_df.copy()
    for key, initiative_name in assortment_to_initiative.items():
        mask = (
            (updated_correct_df[assortment_grp_col] == key[0])
            & (updated_correct_df[email_col] == key[1])
            & missing_initiative_mask
        )
        updated_correct_df.loc[mask, cfg.INITIATIVE_NAME_INPUT_COL] = initiative_name
    # ... (rest of function unchanged, but use key[0] for assortment group and key[1] for email where needed) ...

    # Create Initiative Master Data (only for newly created initiatives)
    initiative_dim_data_records = []
    new_initiatives_mapping = {}

    # Only create master data for newly created initiatives
    if groups_needing_new_initiatives:
        for assortment_group, email in groups_needing_new_initiatives:
            initiative_name = assortment_to_initiative[(assortment_group, email)]
            new_initiatives_mapping[(assortment_group, email)] = initiative_name

            # Get description from the first record of this assortment group
            group_records = missing_initiative_df[
                (missing_initiative_df[assortment_grp_col] == assortment_group)
                & (missing_initiative_df[email_col] == email)
            ]
            if not group_records.empty and initiative_desc_col in group_records.columns:
                description = str(group_records.iloc[0][initiative_desc_col]).strip()
                if not description or description == "nan":
                    description = initiative_name
            else:
                description = initiative_name

            initiative_dim_data_records.append(
                {
                    cfg.INITIATIVE_NAME_OUTPUT_COL: initiative_name,
                    cfg.INITIATIVE_DISPLAY_NAME_COL: initiative_name,
                    cfg.INITIATIVE_DESCRIPTION_COL: description,
                    cfg.INITIATIVE_TYPE_COL: "NPI",
                    cfg.INITIATIVE_LEVEL_8_COL: "NPI",
                    cfg.INITIATIVE_LEVEL_7_COL: "NPI",
                    cfg.INITIATIVE_LEVEL_6_COL: "NPI",
                    cfg.INITIATIVE_LEVEL_5_COL: "NPI",
                    cfg.INITIATIVE_LEVEL_4_COL: "NPI",
                    cfg.INITIATIVE_LEVEL_3_COL: "NPI",
                    cfg.INITIATIVE_LEVEL_2_COL: "NPI",
                    cfg.INITIATIVE_LEVEL_1_COL: "NPI",
                    cfg.INITIATIVE_CATEGORY_COL: "NPI",
                }
            )

    initiative_dim_data = pd.DataFrame(initiative_dim_data_records)

    # Upload initiative dimension data to o9 platform (only for new initiatives)
    if not initiative_dim_data.empty:
        logger.info(
            f"Uploading {len(initiative_dim_data)} NEW initiative dimension records to o9 platform"
        )
        logger.info(
            f"NEW initiative names being uploaded: {initiative_dim_data[cfg.INITIATIVE_NAME_OUTPUT_COL].tolist()}"
        )

        # Perform actual upload to o9 platform
        try:
            create_members(dataframe=initiative_dim_data, logger=logger)
            logger.info("NEW initiative dimension data uploaded successfully to o9 platform")
            logger.info(
                f"These NEW initiatives should now be available in o9: {initiative_dim_data[cfg.INITIATIVE_NAME_OUTPUT_COL].tolist()}"
            )
        except Exception as e:
            logger.error(f"CRITICAL: Failed to upload NEW initiative dimension data: {str(e)}")
            logger.error("This will cause initiative reference errors in valid_entries output")
            logger.error(
                f"Attempted to upload NEW initiatives: {initiative_dim_data[cfg.INITIATIVE_NAME_OUTPUT_COL].tolist()}"
            )
            logger.warning(
                "Continuing processing despite NEW initiative upload failure - expect o9 reference errors"
            )
    else:
        logger.info("No new initiative dimension records to upload - all initiatives already exist")

    # Create Initiative Fact Data
    initiative_fact_data_records = []
    current_date = get_current_date_midnight()

    logger.info("Creating Initiative Fact Data ONLY for newly created initiatives")

    # FIXED: Only create fact data for newly created initiatives, not existing ones
    if groups_needing_new_initiatives:
        for assortment_group, email in groups_needing_new_initiatives:
            initiative_name = assortment_to_initiative[(assortment_group, email)]
            # FIXED: Look at ALL records for this assortment group, not just missing_initiative_df
            group_records = updated_correct_df[
                (updated_correct_df[assortment_grp_col] == assortment_group)
                & (updated_correct_df[email_col] == email)
            ]

            logger.info(
                f"Processing NEW initiative for assortment group {assortment_group}, email {email} with {len(group_records)} total records"
            )

            # Parse launch dates with time set to 00:00:00
            launch_dates = []
            eol_dates = []

            for _, record in group_records.iterrows():
                # Use robust date conversion for launch and EOL dates
                launch_dt = convert_and_validate_date_for_initiative(
                    record.get(launch_date_col, ""), "Product Launch Date"
                )
                eol_dt = convert_and_validate_date_for_initiative(
                    record.get(eol_date_col, ""), "Product EOL Date"
                )

                if launch_dt:
                    launch_dates.append(launch_dt)
                    logger.debug(f"  Found launch date: {launch_dt}")
                if eol_dt:
                    eol_dates.append(eol_dt)
                    logger.debug(f"  Found EOL date: {eol_dt}")

            # Get min launch date and max EOL date
            min_launch_date = min(launch_dates) if launch_dates else None
            max_eol_date = max(eol_dates) if eol_dates else None

            logger.info(
                f"  NEW Initiative {initiative_name} for assortment {assortment_group}, email {email}"
            )
            logger.info(f"  Min launch date: {min_launch_date} (from {len(launch_dates)} dates)")
            logger.info(f"  Max EOL date: {max_eol_date} (from {len(eol_dates)} dates)")

            # Get email from first record
            email = str(group_records.iloc[0].get(email_col, "")).strip()
            version_name = str(group_records.iloc[0].get("Version.[Version Name]", "")).strip()

            initiative_fact_data_records.append(
                {
                    cfg.VERSION_COL: version_name,
                    cfg.INITIATIVE_NAME_OUTPUT_COL: initiative_name,
                    cfg.NPI_INITIATIVE_START_DATE_COL: min_launch_date,
                    cfg.NPI_INITIATIVE_END_DATE_COL: max_eol_date,
                    cfg.LAUNCH_DATE_COL: min_launch_date,
                    cfg.EOL_DATE_COL: max_eol_date,
                    cfg.INITIATIVE_OWNER_COL: email,
                    cfg.NPI_INITIATIVE_CREATED_BY_COL: email,
                    cfg.NPI_INITIATIVE_CREATED_DATE_COL: current_date,
                    cfg.NPI_INITIATIVE_STATUS_COL: "Active",
                }
            )
    else:
        logger.info("No new initiatives created - no initiative fact data to generate")

    initiative_fact_data = pd.DataFrame(initiative_fact_data_records)

    # Format date columns to MM/DD/YYYY string format
    for date_col in [
        cfg.LAUNCH_DATE_COL,
        cfg.EOL_DATE_COL,
        cfg.NPI_INITIATIVE_START_DATE_COL,
        cfg.NPI_INITIATIVE_END_DATE_COL,
        cfg.NPI_INITIATIVE_CREATED_DATE_COL,
    ]:
        if date_col in initiative_fact_data.columns:

            initiative_fact_data[date_col] = initiative_fact_data[date_col].apply(
                format_initiative_date
            )

    # Create Initiative Level Fact Data (only for new initiatives)
    initiative_level_fact_data = create_initiative_level_associations(
        missing_initiative_df,
        new_initiatives_mapping,
        assortment_grp_col,
        email_col,
        cfg.LEVEL_NAME_COL,
        global_npi_levels_df,
    )

    # Create Valid Entries: Final Assortment Data
    valid_entries = create_final_assortment_data(
        updated_correct_df, launch_date_col, eol_date_col, initiative_desc_col
    )

    # Create Initiative Stage Fact File (only for new initiatives)
    initiative_stage_fact_file = create_initiative_stage_fact_file(
        missing_initiative_df,
        new_initiatives_mapping,
        assortment_grp_col,
        email_col,
        data_validation_df,
        cfg.VERSION_COL,
    )

    logger.info("Completed initiative creation")
    return (
        updated_correct_df,
        initiative_dim_data,
        initiative_fact_data,
        initiative_level_fact_data,
        valid_entries,
        initiative_stage_fact_file,
    )


def create_initiative_level_associations(
    missing_initiative_df,
    assortment_to_initiative,
    assortment_grp_col,
    email_col,
    level_name_col,
    global_npi_levels_df,
):
    """Create initiative level associations for newly created initiatives."""
    logger.info("Creating initiative level associations")
    cfg = ValidationConfig

    output3_data = []
    # Create level sequence mapping from GlobalNPILevels
    level_sequence_map = {}
    if not global_npi_levels_df.empty and cfg.npi_level_sequence in global_npi_levels_df.columns:
        for _, row in global_npi_levels_df.iterrows():
            level_name = str(row.get(cfg.DATA_OBJECT_COL, "")).strip()
            level_sequence = row.get("NPI Level Sequence", 0)
            level_sequence_map[level_name] = {
                "sequence": level_sequence,
                "item_level": str(row.get("Global NPI Item Level", "")).strip(),
                "account_level": str(row.get("Global NPI Account Level", "")).strip(),
                "channel_level": str(row.get("Global NPI Channel Level", "")).strip(),
                "region_level": str(row.get("Global NPI Region Level", "")).strip(),
                "pnl_level": str(row.get("Global NPI PnL Level", "")).strip(),
                "demand_domain_level": str(row.get("Global NPI Demand Domain Level", "")).strip(),
                "location_level": str(row.get("Global NPI Location Level", "")).strip(),
                "forecast_method": str(row.get("NPI Forecast Generation Method", "")).strip(),
                "ramp_up_bucket": str(row.get("NPI Ramp Up Bucket", "")).strip(),
                "ramp_up_period": row.get("NPI Ramp Up Period", 0),
            }
    for key, initiative_name in assortment_to_initiative.items():
        assortment_group, email = key
        group_records = missing_initiative_df[
            (missing_initiative_df[assortment_grp_col] == assortment_group)
            & (missing_initiative_df[email_col] == email)
        ]
        if not group_records.empty:
            provided_level = str(group_records.iloc[0].get(level_name_col, "")).strip()
            version_name = str(group_records.iloc[0].get(cfg.VERSION_COL, "")).strip()
            if provided_level in level_sequence_map:
                provided_sequence = level_sequence_map[provided_level]["sequence"]
                applicable_levels = [
                    (level, info)
                    for level, info in level_sequence_map.items()
                    if info["sequence"] >= provided_sequence
                ]
                applicable_levels.sort(key=lambda x: x[1]["sequence"])
                for i, (level_name, level_info) in enumerate(applicable_levels):
                    is_provided_level = level_name == provided_level
                    is_lowest_level = i == len(applicable_levels) - 1
                    output3_data.append(
                        {
                            cfg.VERSION_COL: version_name,
                            cfg.INITIATIVE_NAME_OUTPUT_COL: initiative_name,
                            cfg.DATA_OBJECT_COL: level_name,
                            cfg.NPI_INITIATIVE_LEVEL_ASSOCIATION_COL: 1,
                            cfg.NPI_Inititative_level_status_COL: (
                                "Active" if is_provided_level else "New"
                            ),
                            cfg.NPI_ITEM_LEVEL_COL: level_info["item_level"],
                            cfg.NPI_ACCOUNT_LEVEL_COL: level_info["account_level"],
                            cfg.NPI_CHANNEL_LEVEL_COL: level_info["channel_level"],
                            cfg.NPI_REGION_LEVEL_COL: level_info["region_level"],
                            cfg.NPI_PNL_LEVEL_COL: level_info["pnl_level"],
                            cfg.NPI_DEMAND_DOMAIN_LEVEL_COL: level_info["demand_domain_level"],
                            cfg.NPI_LOCATION_LEVEL_COL: level_info["location_level"],
                            cfg.NPI_fcst_generation_method_l1_COL: level_info["forecast_method"],
                            cfg.NPI_LEVEL_SEQUENCE_COL_L1: level_info["sequence"],
                            cfg.NPI_Ramp_up_bucket_l1_COL: level_info["ramp_up_bucket"],
                            cfg.NPI_Ramp_up_period_l1_COL: level_info["ramp_up_period"],
                            cfg.NPI_Initiative_lowest_level_COL: 1 if is_lowest_level else 0,
                        }
                    )

    return pd.DataFrame(output3_data)


def create_existing_valid_entries(invalid_entries, cfg):
    """Create valid_entries from existing intersections in invalid_entries where NPI Data Rejected Flag = False."""
    logger.info("Creating valid_entries from existing intersections in invalid_entries")

    # Filter for records where NPI Data Rejected Flag = False
    existing_valid_mask = ~invalid_entries[cfg.REJECTED_FLAG_COL]
    existing_valid_df = invalid_entries[existing_valid_mask].copy()

    if existing_valid_df.empty:
        logger.info("No existing valid entries found in invalid_entries")
        return pd.DataFrame(columns=cfg.valid_entries_output_req_cols)

    logger.info(f"Found {len(existing_valid_df)} existing valid entries in invalid_entries")

    # Create valid_entries dataframe with proper column mapping
    valid_entries = existing_valid_df.copy()

    # First, copy values from Original columns to replace the original entity columns
    entity_columns = [
        cfg.NPI_ITEM_COL,
        cfg.NPI_ACCOUNT_COL,
        cfg.NPI_CHANNEL_COL,
        cfg.NPI_REGION_COL,
        cfg.NPI_PNL_COL,
        cfg.NPI_DEMAND_DOMAIN_COL,
        cfg.NPI_LOCATION_COL,
    ]

    for col in entity_columns:
        original_col = col + " Original"
        if original_col in valid_entries.columns:
            # Use the mapped values from the Original column
            logger.debug(f"Using values from '{original_col}' to replace '{col}'")
            valid_entries[col] = valid_entries[original_col]

    # Column mapping for output format
    column_mapping = {
        cfg.VERSION_COL: cfg.VERSION_COL,
        cfg.EMAIL_COL: cfg.EMAIL_COL,
        cfg.INITIATIVE_NAME_INPUT_COL: cfg.INITIATIVE_NAME_OUTPUT_COL,
        cfg.LEVEL_NAME_COL: cfg.DATA_OBJECT_COL,
        cfg.NPI_ITEM_COL: cfg.VALID_ENTRIES_NPI_ITEM_COL,
        cfg.NPI_ACCOUNT_COL: cfg.VALID_ENTRIES_NPI_ACCOUNT_COL,
        cfg.NPI_CHANNEL_COL: cfg.VALID_ENTRIES_NPI_CHANNEL_COL,
        cfg.NPI_REGION_COL: cfg.VALID_ENTRIES_NPI_REGION_COL,
        cfg.NPI_PNL_COL: cfg.VALID_ENTRIES_NPI_PNL_COL,
        cfg.NPI_DEMAND_DOMAIN_COL: cfg.VALID_ENTRIES_NPI_DEMAND_DOMAIN_COL,
        cfg.NPI_LOCATION_COL: cfg.VALID_ENTRIES_NPI_LOCATION_COL,
        cfg.NPI_LAUNCH_DATE_COL: cfg.NPI_LAUNCH_DATE_L0_COL,
        cfg.NPI_EOL_DATE_COL: cfg.NPI_EOL_DATE_L0_COL,
    }

    valid_entries = valid_entries.rename(columns=column_mapping)

    # Remove duplicate columns
    duplicate_cols = valid_entries.columns[valid_entries.columns.duplicated()].tolist()
    if duplicate_cols:
        logger.warning(
            f"Found duplicate columns after mapping: {duplicate_cols} - removing duplicates"
        )
        valid_entries = valid_entries.loc[:, ~valid_entries.columns.duplicated()]

    # Remove temporary "Original" columns
    original_cols_to_remove = [col for col in valid_entries.columns if col.endswith(" Original")]
    if original_cols_to_remove:
        valid_entries.drop(columns=original_cols_to_remove, inplace=True)
        logger.debug(f"Removed temporary original columns: {original_cols_to_remove}")

    # Add NPI Data Accepted Flag = True for existing valid entries
    valid_entries[cfg.NPI_DATA_ACCEPTED_FLAG_COL] = 1

    # Convert date columns to MM/DD/YYYY format string
    for date_col in [cfg.NPI_LAUNCH_DATE_L0_COL, cfg.NPI_EOL_DATE_L0_COL]:
        if date_col in valid_entries.columns:
            logger.info(
                f"Formatting {date_col} in existing valid entries to MM/DD/YYYY string format"
            )

            valid_entries[date_col] = valid_entries[date_col].apply(
                lambda x: convert_and_format_date(x, f"Existing_Valid_Entries {date_col}")
            )

    # Ensure all required columns are present
    required_columns = cfg.valid_entries_output_req_cols
    for col in required_columns:
        if col not in valid_entries.columns:
            valid_entries[col] = ""

    # Safety check: Filter out rows with empty initiative names
    if cfg.INITIATIVE_NAME_OUTPUT_COL in valid_entries.columns:
        empty_initiative_mask = (
            valid_entries[cfg.INITIATIVE_NAME_OUTPUT_COL].isna()
            | (valid_entries[cfg.INITIATIVE_NAME_OUTPUT_COL] == "")
            | (valid_entries[cfg.INITIATIVE_NAME_OUTPUT_COL].astype(str).str.strip() == "")
        )
        if empty_initiative_mask.sum() > 0:
            logger.warning(
                f"Filtering out {empty_initiative_mask.sum()} existing valid entries with empty initiative names"
            )
            valid_entries = valid_entries[~empty_initiative_mask]

    # Reorder columns
    valid_entries = valid_entries[required_columns]

    logger.info(f"Created {len(valid_entries)} existing valid entries for valid_entries output")
    return valid_entries


def create_final_assortment_data(
    updated_correct_df, launch_date_col, eol_date_col, initiative_desc_col
):
    """Create final assortment data output table."""
    logger.info("Creating final assortment data")
    cfg = ValidationConfig

    # Drop unnecessary columns including temporary "Original" columns
    columns_to_drop = []
    if initiative_desc_col in updated_correct_df.columns:
        columns_to_drop.append(initiative_desc_col)
    if "Sequence.[Sequence]" in updated_correct_df.columns:
        columns_to_drop.append("Sequence.[Sequence]")

    # Don't drop Original columns yet - we need them for mapping
    valid_entries = updated_correct_df.drop(columns=columns_to_drop, errors="ignore").copy()

    # First, copy values from Original columns to replace the original entity columns
    entity_columns = [
        cfg.NPI_ITEM_COL,
        cfg.NPI_ACCOUNT_COL,
        cfg.NPI_CHANNEL_COL,
        cfg.NPI_REGION_COL,
        cfg.NPI_PNL_COL,
        cfg.NPI_DEMAND_DOMAIN_COL,
        cfg.NPI_LOCATION_COL,
    ]

    for col in entity_columns:
        original_col = col + " Original"
        if original_col in valid_entries.columns:
            # Use the mapped values from the Original column
            logger.debug(f"Using values from '{original_col}' to replace '{col}'")
            valid_entries[col] = valid_entries[original_col]

    # Debug: Log the column mapping process
    logger.info(f"Before column mapping - columns: {valid_entries.columns.tolist()}")
    if cfg.INITIATIVE_NAME_INPUT_COL in valid_entries.columns:
        logger.info(
            f"Sample initiative names: {valid_entries[cfg.INITIATIVE_NAME_INPUT_COL].head().tolist()}"
        )

    # Now rename columns to match output format - CLEARER: Use explicit input->output mapping
    column_mapping = {
        cfg.VERSION_COL: cfg.VERSION_COL,
        cfg.EMAIL_COL: cfg.EMAIL_COL,
        cfg.INITIATIVE_NAME_INPUT_COL: cfg.INITIATIVE_NAME_OUTPUT_COL,  # Input -> Output: "NPI Initiative Name" -> "Initiative.[Initiative]"
        cfg.LEVEL_NAME_COL: cfg.DATA_OBJECT_COL,
        cfg.NPI_ITEM_COL: cfg.VALID_ENTRIES_NPI_ITEM_COL,
        cfg.NPI_ACCOUNT_COL: cfg.VALID_ENTRIES_NPI_ACCOUNT_COL,
        cfg.NPI_CHANNEL_COL: cfg.VALID_ENTRIES_NPI_CHANNEL_COL,
        cfg.NPI_REGION_COL: cfg.VALID_ENTRIES_NPI_REGION_COL,
        cfg.NPI_PNL_COL: cfg.VALID_ENTRIES_NPI_PNL_COL,
        cfg.NPI_DEMAND_DOMAIN_COL: cfg.VALID_ENTRIES_NPI_DEMAND_DOMAIN_COL,
        cfg.NPI_LOCATION_COL: cfg.VALID_ENTRIES_NPI_LOCATION_COL,
        launch_date_col: cfg.NPI_LAUNCH_DATE_L0_COL,
        eol_date_col: cfg.NPI_EOL_DATE_L0_COL,
    }

    valid_entries = valid_entries.rename(columns=column_mapping)

    # FIRST: Remove duplicate columns immediately after column mapping
    duplicate_cols = valid_entries.columns[valid_entries.columns.duplicated()].tolist()
    if duplicate_cols:
        logger.warning(
            f"Found duplicate columns after mapping: {duplicate_cols} - removing duplicates"
        )
        valid_entries = valid_entries.loc[:, ~valid_entries.columns.duplicated()]
        logger.info(f"After removing duplicates, columns are: {valid_entries.columns.tolist()}")

    # NOW: Safe to access columns for debug logging
    logger.info(
        f"After column mapping and deduplication - columns: {valid_entries.columns.tolist()}"
    )
    if cfg.INITIATIVE_NAME_OUTPUT_COL in valid_entries.columns:
        logger.info(
            f"Sample mapped initiative names: {valid_entries[cfg.INITIATIVE_NAME_OUTPUT_COL].head().tolist()}"
        )

    # Remove temporary "Original" columns that are no longer needed
    original_cols_to_remove = [col for col in valid_entries.columns if col.endswith(" Original")]
    if original_cols_to_remove:
        valid_entries.drop(columns=original_cols_to_remove, inplace=True)
        logger.debug(
            f"Removed temporary original columns from Valid_Entries: {original_cols_to_remove}"
        )

    # Add NPI Data Accepted Flag
    valid_entries[cfg.NPI_DATA_ACCEPTED_FLAG_COL] = 1

    # Convert date columns to MM/DD/YYYY format string
    for date_col in [cfg.NPI_LAUNCH_DATE_L0_COL, cfg.NPI_EOL_DATE_L0_COL]:
        if date_col in valid_entries.columns:
            logger.info(f"Formatting {date_col} in Valid_Entries to MM/DD/YYYY string format")

            valid_entries[date_col] = valid_entries[date_col].apply(
                lambda x: convert_and_format_date(x, f"Valid_Entries {date_col}")
            )

    # Ensure all required columns are present
    required_columns = cfg.valid_entries_output_req_cols

    for col in required_columns:
        if col not in valid_entries.columns:
            valid_entries[col] = ""

    # Safety check: Filter out rows with empty initiative names
    if cfg.INITIATIVE_NAME_OUTPUT_COL in valid_entries.columns:
        empty_initiative_mask = (
            valid_entries[cfg.INITIATIVE_NAME_OUTPUT_COL].isna()
            | (valid_entries[cfg.INITIATIVE_NAME_OUTPUT_COL] == "")
            | (valid_entries[cfg.INITIATIVE_NAME_OUTPUT_COL].astype(str).str.strip() == "")
        )
        if empty_initiative_mask.sum() > 0:
            logger.warning(
                f"Filtering out {empty_initiative_mask.sum()} records with empty initiative names"
            )
            valid_entries = valid_entries[~empty_initiative_mask]

    # Reorder columns
    valid_entries = valid_entries[required_columns]

    return valid_entries


def create_initiative_stage_fact_file(
    missing_initiative_df,
    assortment_to_initiative,
    assortment_grp_col,
    email_col,
    data_validation_df,
    version_col,
):
    """Create initiative stage fact file for stage associations."""
    logger.info("Creating Initiative Stage Fact File")
    cfg = ValidationConfig
    initiative_stage_fact_file_records = []
    current_date = get_current_date_midnight()

    # Get all data validation stages
    if not data_validation_df.empty and len(data_validation_df.columns) > 0:
        data_validation_stages = data_validation_df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        # No data validation stages available - don't create any stage fact records
        logger.info(
            "DataValidation dataframe is empty or has no data - no stage fact records will be created"
        )
        return pd.DataFrame(columns=cfg.initiative_stage_fact_file_req_output_cols)

    logger.info(
        f"Creating stage fact records for {len(data_validation_stages)} stages: {data_validation_stages}"
    )

    # Create stage fact data for each new initiative
    for key, initiative_name in assortment_to_initiative.items():
        assortment_group, email = key  # Unpack the tuple key
        group_records = missing_initiative_df[
            (missing_initiative_df[assortment_grp_col] == assortment_group)
            & (missing_initiative_df[email_col] == email)
        ]

        if not group_records.empty:
            # Get email from the first record of this assortment group
            first_record = group_records.iloc[0]
            email = first_record.get(email_col, "")
            version_name = first_record.get(version_col, "")  # Get actual version from FileUpload

            # Create stage fact record for EACH stage
            for stage in data_validation_stages:
                stage_record = {
                    version_col: version_name,  # Use actual version from FileUpload
                    cfg.INITIATIVE_NAME_OUTPUT_COL: initiative_name,
                    cfg.Data_validation_COL: stage,  # Each stage (Stage 1, Stage 2, Stage 3, Stage 4, Stage 5)
                    cfg.Stage_association_COL: 1,
                    cfg.Stage_owner_COL: email,
                    cfg.Stage_created_date_COL: format_stage_date_to_mmddyyyy_with_time(
                        current_date
                    ),
                }

                initiative_stage_fact_file_records.append(stage_record)
                logger.debug(
                    f"Created stage fact record for initiative {initiative_name} with stage {stage}"
                )

            logger.info(
                f"Created {len(data_validation_stages)} stage fact records for initiative {initiative_name}"
            )

    # Create DataFrame
    if initiative_stage_fact_file_records:
        initiative_stage_fact_file = pd.DataFrame(initiative_stage_fact_file_records)
        logger.info(
            f"Created {len(initiative_stage_fact_file)} total initiative stage fact file records"
        )
    else:
        initiative_stage_fact_file = pd.DataFrame(
            columns=cfg.initiative_stage_fact_file_req_output_cols
        )
        logger.info("No initiative stage fact file records to create")
    return initiative_stage_fact_file


def check_npi_assortment_association(
    remarktable,
    assortment_grp_col,
    initiative_name_col,
    level_name_col,
    npi_item_name_col,
    npi_account_name_col,
    npi_channel_name_col,
    npi_region_name_col,
    npi_pnl_name_col,
    npi_demand_domain_name_col,
    npi_location_name_col,
    rejected_flag_col,
    validation_remark_col,
    npi_assortments_df,
):
    """Check if the intersection/assortment is already linked with another initiative."""
    logger.info(
        "Starting validation check: NPIAssortment association validation (skipping already rejected records)"
    )

    if npi_assortments_df.empty:
        logger.info("NPIAssortments dataframe is empty - skipping association validation")
        return remarktable

    # Required columns in NPIAssortments
    required_cols = cfg.NPI_ASSORTMENTS_REQUIRED_COLS

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in npi_assortments_df.columns]
    if missing_cols:
        logger.warning(
            f"Missing columns in NPIAssortments: {missing_cols} - skipping association validation"
        )
        return remarktable

    # Filter NPIAssortments for active associations (NPI Association L0 = 1)
    active_assortments = npi_assortments_df[npi_assortments_df["NPI Association L0"] == 1].copy()

    if active_assortments.empty:
        logger.info("No active associations found in NPIAssortments - skipping validation")
        return remarktable

    logger.info(f"Found {len(active_assortments)} active associations in NPIAssortments")

    # Process each row in remarktable
    for idx, row in remarktable.iterrows():
        # Skip validation if NPI Data Rejected Flag is True
        if row[rejected_flag_col]:
            logger.debug(
                f"Row {idx}: Skipping NPIAssortment association validation - already rejected"
            )
            continue

        # Get intersection details from remarktable using Original columns (after DisplayName mapping)
        initiative_name = (
            str(row[initiative_name_col]).strip() if pd.notna(row[initiative_name_col]) else ""
        )
        # level_name = str(row[level_name_col]).strip() if pd.notna(row[level_name_col]) else ""

        # Use Original columns if they exist, otherwise fall back to regular columns
        item_name_col_original = f"{npi_item_name_col} Original"
        account_name_col_original = f"{npi_account_name_col} Original"
        channel_name_col_original = f"{npi_channel_name_col} Original"
        region_name_col_original = f"{npi_region_name_col} Original"
        pnl_name_col_original = f"{npi_pnl_name_col} Original"
        demand_domain_name_col_original = f"{npi_demand_domain_name_col} Original"
        location_name_col_original = f"{npi_location_name_col} Original"

        # Use Original columns if available, otherwise use regular columns
        item_name = (
            str(row[item_name_col_original]).strip()
            if item_name_col_original in remarktable.columns
            and pd.notna(row[item_name_col_original])
            else str(row[npi_item_name_col]).strip() if pd.notna(row[npi_item_name_col]) else ""
        )
        account_name = (
            str(row[account_name_col_original]).strip()
            if account_name_col_original in remarktable.columns
            and pd.notna(row[account_name_col_original])
            else (
                str(row[npi_account_name_col]).strip()
                if pd.notna(row[npi_account_name_col])
                else ""
            )
        )
        channel_name = (
            str(row[channel_name_col_original]).strip()
            if channel_name_col_original in remarktable.columns
            and pd.notna(row[channel_name_col_original])
            else (
                str(row[npi_channel_name_col]).strip()
                if pd.notna(row[npi_channel_name_col])
                else ""
            )
        )
        region_name = (
            str(row[region_name_col_original]).strip()
            if region_name_col_original in remarktable.columns
            and pd.notna(row[region_name_col_original])
            else str(row[npi_region_name_col]).strip() if pd.notna(row[npi_region_name_col]) else ""
        )
        pnl_name = (
            str(row[pnl_name_col_original]).strip()
            if pnl_name_col_original in remarktable.columns and pd.notna(row[pnl_name_col_original])
            else str(row[npi_pnl_name_col]).strip() if pd.notna(row[npi_pnl_name_col]) else ""
        )
        demand_domain_name = (
            str(row[demand_domain_name_col_original]).strip()
            if demand_domain_name_col_original in remarktable.columns
            and pd.notna(row[demand_domain_name_col_original])
            else (
                str(row[npi_demand_domain_name_col]).strip()
                if pd.notna(row[npi_demand_domain_name_col])
                else ""
            )
        )
        location_name = (
            str(row[location_name_col_original]).strip()
            if location_name_col_original in remarktable.columns
            and pd.notna(row[location_name_col_original])
            else (
                str(row[npi_location_name_col]).strip()
                if pd.notna(row[npi_location_name_col])
                else ""
            )
        )

        # Find matching intersection in NPIAssortments
        matching_assortments = active_assortments[
            (active_assortments["Item.[NPI Item]"].astype(str).str.strip() == item_name)
            & (active_assortments["Account.[NPI Account]"].astype(str).str.strip() == account_name)
            & (active_assortments["Channel.[NPI Channel]"].astype(str).str.strip() == channel_name)
            & (active_assortments["Region.[NPI Region]"].astype(str).str.strip() == region_name)
            & (active_assortments["PnL.[NPI PnL]"].astype(str).str.strip() == pnl_name)
            & (
                active_assortments["Demand Domain.[NPI Demand Domain]"].astype(str).str.strip()
                == demand_domain_name
            )
            & (
                active_assortments["Location.[NPI Location]"].astype(str).str.strip()
                == location_name
            )
        ]

        if not matching_assortments.empty:
            # Found matching intersection in NPIAssortments
            existing_initiative = str(
                matching_assortments.iloc[0]["Initiative.[Initiative]"]
            ).strip()

            if initiative_name:
                # Case I: Initiative name is provided
                if initiative_name == existing_initiative:
                    # Same initiative - validation passes
                    logger.debug(
                        f"Row {idx}: Initiative '{initiative_name}' matches existing association"
                    )
                    continue
                else:
                    # Case II: Different initiative - reject
                    logger.info(
                        f"Row {idx}: Initiative '{initiative_name}' conflicts with existing association '{existing_initiative}'"
                    )
                    remarktable.loc[idx, rejected_flag_col] = True
                    remark = f"Provided assortment is already associated with {existing_initiative} initiative."
                    append_validation_remark_vectorized(
                        remarktable, pd.Series([True], index=[idx]), validation_remark_col, remark
                    )
            else:
                # Case II: Initiative name is empty but intersection exists - reject
                logger.info(
                    f"Row {idx}: Empty initiative name but intersection already associated with '{existing_initiative}'"
                )
                remarktable.loc[idx, rejected_flag_col] = True
                remark = f"Provided assortment is already associated with {existing_initiative} initiative."
                append_validation_remark_vectorized(
                    remarktable, pd.Series([True], index=[idx]), validation_remark_col, remark
                )

    logger.info(
        "Completed validation check: NPIAssortment association validation (skipped already rejected records)"
    )
    return remarktable


def propagate_group_rejections(remarktable, assortment_grp_col, rejected_flag_col):
    """Propagate rejections to all records in the same assortment group."""
    logger.info("Propagating rejections to all records in the same assortment group")

    # Find all rejected records
    rejected_mask = remarktable[rejected_flag_col].astype(bool)

    if rejected_mask.sum() == 0:
        logger.info("No rejected records found - no propagation needed")
        return remarktable

    # Get all assortment groups that have at least one rejected record
    rejected_groups = remarktable.loc[rejected_mask, assortment_grp_col].unique()

    logger.info(f"Found {len(rejected_groups)} assortment groups with rejected records")

    # For each rejected group, mark all records in that group as rejected
    for group in rejected_groups:
        group_mask = remarktable[assortment_grp_col] == group
        remarktable.loc[group_mask, rejected_flag_col] = True

        # Add propagation remark to records that weren't originally rejected
        newly_rejected = remarktable.loc[group_mask & ~rejected_mask, :]

        if not newly_rejected.empty:
            append_validation_remark_vectorized(
                remarktable,
                group_mask & ~rejected_mask,
                remarktable.columns[remarktable.columns.str.contains("Validation Remark")][0],
                "Record rejected due to other records in the same assortment group being rejected",
            )

    logger.info(f"Propagated rejections to all records in {len(rejected_groups)} assortment groups")
    return remarktable


def run_level_name_validation(
    invalid_entries, level_name_col, rejected_flag_col, validation_remark_col, npi_level_df
):
    """Run NPI Level Name validity check against NPILevel dataframe (including display names)."""
    logger.info("Running NPI Level Name validity check")

    # Pre-compute valid NPI levels (including display names)
    valid_npi_levels = set()
    level_display_name_mapping = {}

    if not npi_level_df.empty:
        level_col = cfg.DATA_OBJECT_COL
        level_display_name_col = cfg.DATA_OBJECT_DISPLAYNAME_COL

        if level_col in npi_level_df.columns:
            # Add all level names from the main column
            valid_npi_levels.update(npi_level_df[level_col].dropna().astype(str).str.strip())

            # Create mapping from display names to level names
            if level_display_name_col in npi_level_df.columns:
                for _, row in npi_level_df.iterrows():
                    level_name = str(row[level_col]).strip()
                    display_name = str(row[level_display_name_col]).strip()

                    if level_name and display_name:
                        # Add display name to valid levels set
                        valid_npi_levels.add(display_name)
                        # Create mapping from display name to level name
                        level_display_name_mapping[display_name] = level_name

                logger.info(
                    f"Found {len(level_display_name_mapping)} level display name to level name mappings"
                )
            else:
                logger.warning(
                    f"Display name column '{level_display_name_col}' not found in NPILevel dataframe"
                )

        logger.info(
            f"Found {len(valid_npi_levels)} valid levels (including display names) in NPILevel dataframe"
        )
    else:
        logger.warning("NPILevel dataframe is empty - no level validation will be performed")
        return invalid_entries

    # Apply level display name to level name replacement for invalid entries
    if level_display_name_mapping:
        # Find entries that have display names and replace them with level names
        level_name_mask = (
            invalid_entries[level_name_col]
            .astype(str)
            .str.strip()
            .isin(level_display_name_mapping.keys())
        )

        if level_name_mask.sum() > 0:
            logger.info(
                f"Found {level_name_mask.sum()} entries with level display names that will be replaced with level names"
            )

            # Create a copy of the column for replacement and handle NaN values
            invalid_entries[level_name_col] = invalid_entries[level_name_col].fillna("").astype(str)

            # Replace display names with level names
            for display_name, level_name in level_display_name_mapping.items():
                mask = invalid_entries[level_name_col].str.strip() == display_name
                if mask.sum() > 0:
                    invalid_entries.loc[mask, level_name_col] = level_name
                    logger.info(
                        f"Replaced {mask.sum()} occurrences of level display name '{display_name}' with level name '{level_name}'"
                    )

    # Validate level names (only for non-empty levels)
    if valid_npi_levels:
        # Create mask for non-empty levels
        non_empty_level_mask = (
            invalid_entries[level_name_col].notna()
            & (invalid_entries[level_name_col] != "")
            & (invalid_entries[level_name_col].astype(str).str.strip() != "")
        )

        if non_empty_level_mask.sum() > 0:
            level_values = (
                invalid_entries.loc[non_empty_level_mask, level_name_col].astype(str).str.strip()
            )
            invalid_level_mask = non_empty_level_mask & ~level_values.isin(valid_npi_levels)
            if invalid_level_mask.sum() > 0:
                append_validation_remark_vectorized(
                    invalid_entries,
                    invalid_level_mask,
                    validation_remark_col,
                    "Level name doesn't exists",
                )
                invalid_entries.loc[invalid_level_mask, rejected_flag_col] = True
                logger.info(
                    f"Marked {invalid_level_mask.sum()} records as rejected for invalid level names"
                )
            else:
                logger.info(
                    "All level names are valid (including display names that were replaced)"
                )
        else:
            logger.info("No non-empty level names to validate")
    else:
        logger.warning("No valid levels found in NPILevel dataframe - level validation skipped")

    return invalid_entries


def run_entity_empty_checks(invalid_entries, rejected_flag_col, validation_remark_col):
    """Run entity empty checks for all entity columns."""
    logger.info("Running entity empty checks")

    # Entity columns for validation
    entity_columns = [
        (cfg.NPI_ITEM_COL, "Item"),
        (cfg.NPI_ACCOUNT_COL, "Account"),
        (cfg.NPI_CHANNEL_COL, "Channel"),
        (cfg.NPI_REGION_COL, "Region"),
        (cfg.NPI_PNL_COL, "PnL"),
        (cfg.NPI_DEMAND_DOMAIN_COL, "DemandDomain"),
        (cfg.NPI_LOCATION_COL, "Location"),
    ]

    for col_name, entity_name in entity_columns:
        if col_name in invalid_entries.columns:
            empty_entity_mask = (
                invalid_entries[col_name].isna()
                | (invalid_entries[col_name] == "")
                | (invalid_entries[col_name].astype(str).str.strip() == "")
            )
            if empty_entity_mask.sum() > 0:
                remark = f"{entity_name} name is required"
                append_validation_remark_vectorized(
                    invalid_entries, empty_entity_mask, validation_remark_col, remark
                )
                invalid_entries.loc[empty_entity_mask, rejected_flag_col] = True

    return invalid_entries


def run_date_validation(
    invalid_entries, launch_date_col, eol_date_col, rejected_flag_col, validation_remark_col
):
    """Run date validation for Launch Date and EOL Date."""
    logger.info("Running date validation for Launch Date and EOL Date")

    # Date validation for Launch Date and EOL Date - check if values are valid dates (not just strings)
    date_columns = [
        (launch_date_col, "NPI Product Launch Date"),
        (eol_date_col, "NPI Product EOL Date"),
    ]

    # Individual date validation with specific error messages
    for date_col, date_name in date_columns:
        if date_col in invalid_entries.columns:
            # Only validate non-empty values
            non_empty_date_mask = (
                invalid_entries[date_col].notna()
                & (invalid_entries[date_col] != "")
                & (invalid_entries[date_col].astype(str).str.strip() != "")
                & (invalid_entries[date_col].astype(str).str.strip() != "nan")
            )

            if non_empty_date_mask.sum() > 0:
                # Check if non-empty values are valid dates

                # Apply date validation only to non-empty values
                non_empty_values = invalid_entries.loc[non_empty_date_mask, date_col]
                valid_dates_series = non_empty_values.apply(is_valid_date)

                # Create a full-sized boolean series for valid dates
                valid_dates_full = pd.Series(True, index=invalid_entries.index)
                valid_dates_full.loc[non_empty_date_mask] = valid_dates_series

                # Create mask for invalid dates (non-empty but unparseable)
                invalid_date_mask = non_empty_date_mask & ~valid_dates_full

                # Apply specific error message for this date column
                if invalid_date_mask.sum() > 0:
                    error_message = f"{date_name.replace('NPI ', '')} format needs to be corrected"
                    append_validation_remark_vectorized(
                        invalid_entries, invalid_date_mask, validation_remark_col, error_message
                    )
                    invalid_entries.loc[invalid_date_mask, rejected_flag_col] = True
                    logger.info(f"Found {invalid_date_mask.sum()} invalid {date_name} values")

    return invalid_entries


def run_initiative_consistency_check(
    invalid_entries,
    assortment_grp_col,
    initiative_name_col,
    rejected_flag_col,
    validation_remark_col,
):
    """Run initiative name consistency check within (assortment group, email) intersections."""
    logger.info("Running initiative name consistency check")
    cfg = ValidationConfig
    empty_assortment_mask = (
        invalid_entries[assortment_grp_col].isna()
        | (invalid_entries[assortment_grp_col] == "")
        | (invalid_entries[assortment_grp_col].astype(str).str.strip() == "")
        | invalid_entries[cfg.EMAIL_COL].isna()
        | (invalid_entries[cfg.EMAIL_COL] == "")
        | (invalid_entries[cfg.EMAIL_COL].astype(str).str.strip() == "")
    )
    if not empty_assortment_mask.all():
        valid_data = invalid_entries[~empty_assortment_mask]
        if not valid_data.empty:
            grouped = valid_data.groupby([assortment_grp_col, cfg.EMAIL_COL])
            for (assortment_group, email), group_df in grouped:
                group_indices = group_df.index
                all_initiative_names = (
                    group_df[initiative_name_col].fillna("").astype(str).str.strip()
                )
                non_empty_initiative_names = all_initiative_names[all_initiative_names != ""]
                empty_initiative_names = all_initiative_names[all_initiative_names == ""]
                unique_non_empty_initiatives = (
                    non_empty_initiative_names.unique()
                    if len(non_empty_initiative_names) > 0
                    else []
                )
                has_blanks = len(empty_initiative_names) > 0
                has_multiple_non_blank = len(unique_non_empty_initiatives) > 1
                has_mixed = len(unique_non_empty_initiatives) == 1 and has_blanks
                if has_multiple_non_blank or has_mixed:
                    inconsistent_mask = invalid_entries.index.isin(group_indices)
                    append_validation_remark_vectorized(
                        invalid_entries,
                        inconsistent_mask,
                        validation_remark_col,
                        'The same "Assortment Group Number" has a different Initiative Name mentioned or is blank',
                    )
                    invalid_entries.loc[inconsistent_mask, rejected_flag_col] = True
                    logger.info(
                        f"Assortment group '{assortment_group}', email '{email}' inconsistency: Multiple different initiative names or mix of blank and non-blank found: {list(unique_non_empty_initiatives)}"
                    )
                    logger.info(
                        f"Marked {inconsistent_mask.sum()} records as rejected for initiative name inconsistency"
                    )
    return invalid_entries


def run_initiative_validity_check(
    invalid_entries,
    initiative_name_col,
    rejected_flag_col,
    validation_remark_col,
    npi_initiative_df,
):
    """Run initiative name validity check against NPIInitiative dataframe (including display names)."""
    logger.info("Running initiative name validity check")

    # Create comprehensive valid initiatives lookup including display names
    valid_initiatives = set()
    display_name_to_initiative_mapping = {}

    if not npi_initiative_df.empty:
        # Check both Initiative.[Initiative] and Initiative.[Initiative$DisplayName] columns
        initiative_col = cfg.INITIATIVE_COL
        display_name_col = cfg.INITIATIVE_DISPLAY_NAME_COL

        if initiative_col in npi_initiative_df.columns:
            # Add all initiative names from the main column
            valid_initiatives.update(
                npi_initiative_df[initiative_col].dropna().astype(str).str.strip()
            )

            # Create mapping from display names to initiative names
            if display_name_col in npi_initiative_df.columns:
                for _, row in npi_initiative_df.iterrows():
                    initiative_name = str(row[initiative_col]).strip()
                    display_name = str(row[display_name_col]).strip()

                    if initiative_name and display_name:
                        # Add display name to valid initiatives set
                        valid_initiatives.add(display_name)
                        # Create mapping from display name to initiative name
                        display_name_to_initiative_mapping[display_name] = initiative_name

                logger.info(
                    f"Found {len(display_name_to_initiative_mapping)} display name to initiative mappings"
                )
            else:
                logger.warning(
                    f"Display name column '{display_name_col}' not found in NPIInitiative dataframe"
                )

        logger.info(
            f"Found {len(valid_initiatives)} valid initiatives (including display names) in NPIInitiative dataframe"
        )
    else:
        logger.warning(
            "NPIInitiative dataframe is empty - no initiative validation will be performed"
        )
        return invalid_entries

    # Apply display name to initiative name replacement for invalid entries
    if display_name_to_initiative_mapping:
        # Find entries that have display names and replace them with initiative names
        initiative_name_mask = (
            invalid_entries[initiative_name_col]
            .astype(str)
            .str.strip()
            .isin(display_name_to_initiative_mapping.keys())
        )

        if initiative_name_mask.sum() > 0:
            logger.info(
                f"Found {initiative_name_mask.sum()} entries with display names that will be replaced with initiative names"
            )

            # Create a copy of the column for replacement and handle NaN values
            invalid_entries[initiative_name_col] = (
                invalid_entries[initiative_name_col].fillna("").astype(str)
            )

            # Replace display names with initiative names
            for display_name, initiative_name in display_name_to_initiative_mapping.items():
                mask = invalid_entries[initiative_name_col].str.strip() == display_name
                if mask.sum() > 0:
                    invalid_entries.loc[mask, initiative_name_col] = initiative_name
                    logger.info(
                        f"Replaced {mask.sum()} occurrences of display name '{display_name}' with initiative name '{initiative_name}'"
                    )

    # Validate initiative names (only for non-empty initiatives)
    if valid_initiatives:
        # Create mask for non-empty initiatives
        non_empty_initiative_mask = (
            invalid_entries[initiative_name_col].notna()
            & (invalid_entries[initiative_name_col] != "")
            & (invalid_entries[initiative_name_col].astype(str).str.strip() != "")
        )

        if non_empty_initiative_mask.sum() > 0:
            initiative_values = (
                invalid_entries.loc[non_empty_initiative_mask, initiative_name_col]
                .astype(str)
                .str.strip()
            )
            invalid_initiative_mask = non_empty_initiative_mask & ~initiative_values.isin(
                valid_initiatives
            )
            if invalid_initiative_mask.sum() > 0:
                append_validation_remark_vectorized(
                    invalid_entries,
                    invalid_initiative_mask,
                    validation_remark_col,
                    "The Initiative Name doesn't exists",
                )
                invalid_entries.loc[invalid_initiative_mask, rejected_flag_col] = True
                logger.info(
                    f"Marked {invalid_initiative_mask.sum()} records as rejected for invalid initiative names"
                )
            else:
                logger.info(
                    "All initiative names are valid (including display names that were replaced)"
                )
        else:
            logger.info("No non-empty initiative names to validate")
    else:
        logger.warning(
            "No valid initiatives found in NPIInitiative dataframe - initiative validation skipped"
        )

    return invalid_entries


def run_initiative_status_check(
    invalid_entries,
    initiative_name_col,
    rejected_flag_col,
    validation_remark_col,
    npi_initiative_status_df,
):
    """Run initiative status check from NPIInitiativeStatus."""
    logger.info("Running initiative status check")

    # Create initiative status lookup from NPIInitiativeStatus
    initiative_status_lookup = {}
    if (
        not npi_initiative_status_df.empty
        and "Initiative.[Initiative]" in npi_initiative_status_df.columns
        and "NPI Initiative Status" in npi_initiative_status_df.columns
    ):
        for _, row in npi_initiative_status_df.iterrows():
            initiative_name = str(row["Initiative.[Initiative]"]).strip()
            initiative_status = str(row["NPI Initiative Status"]).strip().lower()
            if initiative_name and initiative_status:
                initiative_status_lookup[initiative_name] = initiative_status
        logger.info(
            f"Found {len(initiative_status_lookup)} initiative statuses in NPIInitiativeStatus dataframe"
        )
    else:
        logger.warning(
            "NPIInitiativeStatus dataframe is empty or missing required columns - no initiative status validation will be performed"
        )
        return invalid_entries

    # Check initiative status for valid initiatives
    cfg = ValidationConfig
    validation_count = 0

    for idx, row in invalid_entries.iterrows():
        # Skip if already rejected
        if row[rejected_flag_col]:
            continue

        initiative_name = str(row.get(initiative_name_col, "")).strip()

        # Skip if initiative name is empty
        if not initiative_name or initiative_name.lower() in ["nan", "none", "null", ""]:
            continue

        # Check if this initiative has invalid status
        if initiative_name in initiative_status_lookup:
            initiative_status = initiative_status_lookup[initiative_name]

            if initiative_status in cfg.INVALID_INITIATIVE_STATUSES:
                logger.info(
                    f"Initiative '{initiative_name}' has invalid status: '{initiative_status}' - rejecting record"
                )
                invalid_entries.loc[idx, rejected_flag_col] = True
                append_validation_remark_vectorized(
                    invalid_entries,
                    pd.Series([True], index=[idx]),
                    validation_remark_col,
                    f"Provided Initiative is {initiative_status}",
                )
                validation_count += 1
        else:
            logger.warning(
                f"Initiative '{initiative_name}' not found in NPIInitiativeStatus dataframe - status validation skipped"
            )

    logger.info(f"Completed initiative status check: {validation_count} records rejected")
    return invalid_entries


def run_initiative_level_status_check(
    invalid_entries,
    initiative_name_col,
    rejected_flag_col,
    validation_remark_col,
    npi_initiative_level_df,
):
    """Run initiative level status check from NPIInitiativeLevel."""
    logger.info("Running initiative level status check")

    # Create initiative level status lookup from NPIInitiativeLevel
    initiative_level_status_lookup = {}
    if (
        not npi_initiative_level_df.empty
        and "Initiative.[Initiative]" in npi_initiative_level_df.columns
        and "NPI Initiative Level Status" in npi_initiative_level_df.columns
    ):
        for _, row in npi_initiative_level_df.iterrows():
            initiative_name = str(row["Initiative.[Initiative]"]).strip()
            initiative_level_status = str(row["NPI Initiative Level Status"]).strip().lower()
            if initiative_name and initiative_level_status:
                initiative_level_status_lookup[initiative_name] = initiative_level_status
        logger.info(
            f"Found {len(initiative_level_status_lookup)} initiative level statuses in NPIInitiativeLevel dataframe"
        )
    else:
        logger.warning(
            "NPIInitiativeLevel dataframe is empty or missing required columns - no initiative level status validation will be performed"
        )
        return invalid_entries

    # Check initiative level status for valid initiatives
    cfg = ValidationConfig
    validation_count = 0

    for idx, row in invalid_entries.iterrows():
        # Skip if already rejected
        if row[rejected_flag_col]:
            continue

        initiative_name = str(row.get(initiative_name_col, "")).strip()

        # Skip if initiative name is empty
        if not initiative_name or initiative_name.lower() in ["nan", "none", "null", ""]:
            continue

        # Check if this initiative has invalid level status
        if initiative_name in initiative_level_status_lookup:
            initiative_level_status = initiative_level_status_lookup[initiative_name]

            if initiative_level_status in cfg.INVALID_INITIATIVE_STATUSES:
                logger.info(
                    f"Initiative '{initiative_name}' has invalid level status: '{initiative_level_status}' - rejecting record"
                )
                invalid_entries.loc[idx, rejected_flag_col] = True
                append_validation_remark_vectorized(
                    invalid_entries,
                    pd.Series([True], index=[idx]),
                    validation_remark_col,
                    f"Provided Initiative Level is {initiative_level_status}",
                )
                validation_count += 1
        else:
            logger.warning(
                f"Initiative '{initiative_name}' not found in NPIInitiativeLevel dataframe - level status validation skipped"
            )

    logger.info(f"Completed initiative level status check: {validation_count} records rejected")
    return invalid_entries


def run_initiative_description_consistency_check(
    invalid_entries,
    assortment_grp_col,
    initiative_name_col,
    initiative_desc_col,
    rejected_flag_col,
    validation_remark_col,
):
    """Run initiative description consistency check when initiative names are blank, per (assortment group, email) intersection."""
    logger.info("Running initiative description consistency check")
    cfg = ValidationConfig
    empty_assortment_mask = (
        invalid_entries[assortment_grp_col].isna()
        | (invalid_entries[assortment_grp_col] == "")
        | (invalid_entries[assortment_grp_col].astype(str).str.strip() == "")
        | invalid_entries[cfg.EMAIL_COL].isna()
        | (invalid_entries[cfg.EMAIL_COL] == "")
        | (invalid_entries[cfg.EMAIL_COL].astype(str).str.strip() == "")
    )
    if not empty_assortment_mask.all():
        valid_data = invalid_entries[~empty_assortment_mask]
        if not valid_data.empty:
            grouped = valid_data.groupby([assortment_grp_col, cfg.EMAIL_COL])
            for (assortment_group, email), group_df in grouped:
                group_indices = group_df.index
                blank_initiative_mask = (
                    group_df[initiative_name_col].isna()
                    | (group_df[initiative_name_col] == "")
                    | (group_df[initiative_name_col].astype(str).str.strip() == "")
                )
                if blank_initiative_mask.sum() > 0:
                    descriptions = group_df[initiative_desc_col].fillna("").astype(str).str.strip()
                    unique_descriptions = descriptions.unique()
                    if len(unique_descriptions) > 1:
                        inconsistent_mask = invalid_entries.index.isin(group_indices)
                        append_validation_remark_vectorized(
                            invalid_entries,
                            inconsistent_mask,
                            validation_remark_col,
                            'The same "Assortment Group Number" has a different Initiative Description',
                        )
                        invalid_entries.loc[inconsistent_mask, rejected_flag_col] = True
                        logger.info(
                            f"Initiative description inconsistency found for assortment group '{assortment_group}', email '{email}' when blank initiative names present"
                        )
                        logger.info(f"Descriptions found: {list(unique_descriptions)}")
                        logger.info(
                            f"Marked {inconsistent_mask.sum()} records as rejected for initiative description inconsistency"
                        )
    return invalid_entries


def run_mandatory_entity_validation(
    invalid_entries,
    level_name_col,
    npi_item_col,
    npi_account_col,
    npi_channel_col,
    npi_region_col,
    npi_pnl_col,
    npi_demand_domain_col,
    npi_location_col,
    rejected_flag_col,
    validation_remark_col,
    global_npi_levels_df,
    master_dfs_config,
):
    """Run mandatory entity validation (always runs regardless of rejection status)."""
    logger.info("Running MANDATORY Entity validation (always runs regardless of rejection status)")

    # Pre-compute all lookups once
    level_mappings = create_entity_level_mappings(global_npi_levels_df)
    master_lookups = create_master_data_lookups(master_dfs_config)

    # Entity validation using vectorized operations
    entity_columns = [
        (npi_item_col, "Item"),
        (npi_account_col, "Account"),
        (npi_channel_col, "Channel"),
        (npi_region_col, "Region"),
        (npi_pnl_col, "PnL"),
        (npi_demand_domain_col, "DemandDomain"),
        (npi_location_col, "Location"),
    ]

    validation_count = 0

    # Ensure rejected_flag_col is boolean
    if invalid_entries[rejected_flag_col].dtype != "bool":
        invalid_entries[rejected_flag_col] = (
            invalid_entries[rejected_flag_col].fillna(False).astype(bool)
        )

    # Batch entity validation by level (MANDATORY - runs for all records regardless of rejection status)
    for npi_level, level_config in level_mappings.items():
        level_mask = invalid_entries[level_name_col].astype(str).str.strip() == npi_level
        if level_mask.sum() > 0:

            for col_name, entity_key in entity_columns:
                if col_name in invalid_entries.columns and entity_key in level_config:
                    # Use original column if available
                    original_col_name = f"{col_name} Original"
                    actual_col_name = (
                        original_col_name
                        if original_col_name in invalid_entries.columns
                        else col_name
                    )

                    # Vectorized entity validation (MANDATORY - no rejection status check)
                    valid_count = validate_entities_vectorized(
                        invalid_entries,
                        level_mask,
                        actual_col_name,
                        entity_key,
                        level_config[entity_key],
                        master_lookups,
                        rejected_flag_col,
                        validation_remark_col,
                    )
                    validation_count += valid_count

    logger.info(f"Completed MANDATORY Entity validation: {validation_count} issues found")
    return invalid_entries


def check_npi_initiative_level_association(
    remarktable,
    initiative_name_col,
    level_name_col,
    rejected_flag_col,
    validation_remark_col,
    npi_initiative_level_df,
):
    """
    Check if the NPI initiative is associated with the provided level.

    Args:
        remarktable (pd.DataFrame): The dataframe to validate
        initiative_name_col (str): Column name for initiative name
        level_name_col (str): Column name for level name
        rejected_flag_col (str): Column name for rejection flag
        validation_remark_col (str): Column name for validation remarks
        npi_initiative_level_df (pd.DataFrame): NPIInitiativeLevel dataframe

    Returns:
        pd.DataFrame: Updated dataframe with validation results
    """
    logger.info("Starting validation check: NPI Initiative Level Association validation")

    if npi_initiative_level_df.empty:
        logger.info("NPIInitiativeLevel dataframe is empty - skipping association validation")
        return remarktable

    # Required columns in NPIInitiativeLevel
    required_cols = cfg.NPI_INITIATIVE_LEVEL_REQUIRED_COLS

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in npi_initiative_level_df.columns]
    if missing_cols:
        logger.warning(
            f"Missing columns in NPIInitiativeLevel: {missing_cols} - skipping association validation"
        )
        return remarktable

    # Filter NPIInitiativeLevel for active associations (NPI Initiative Level Association = 1)
    active_associations = npi_initiative_level_df[
        npi_initiative_level_df["NPI Initiative Level Association"] == 1
    ].copy()

    if active_associations.empty:
        logger.info("No active associations found in NPIInitiativeLevel - skipping validation")
        return remarktable

    # Create a lookup set for faster validation
    association_lookup = set()
    for _, row in active_associations.iterrows():
        initiative_name = str(row[cfg.INITIATIVE_NAME_OUTPUT_COL]).strip()
        level_name = str(row[cfg.DATA_OBJECT_COL]).strip()
        if initiative_name and level_name:
            association_lookup.add((initiative_name, level_name))

    logger.info(
        f"Created association lookup with {len(association_lookup)} valid initiative-level combinations"
    )

    # Validate each row - only for rows where initiative name is actually defined
    validation_count = 0
    for idx, row in remarktable.iterrows():
        # Skip if already rejected by previous validations
        if row[rejected_flag_col]:
            logger.debug(
                f"Row {idx}: Skipping NPI Initiative Level Association validation - already rejected"
            )
            continue

        initiative_name = str(row.get(initiative_name_col, "")).strip()
        level_name = str(row.get(level_name_col, "")).strip()

        # Skip if initiative name is empty, null, or nan - only check rows with defined initiative names
        if not initiative_name or initiative_name.lower() in ["nan", "none", "null", ""]:
            logger.debug(f"Row {idx}: Skipping validation - initiative name is empty/null/nan")
            continue

        # Skip if level name is empty
        if not level_name:
            logger.debug(f"Row {idx}: Skipping validation - level name is empty")
            continue

        # Check if this initiative-level combination is associated
        if (initiative_name, level_name) not in association_lookup:
            logger.info(
                f"Row {idx}: Initiative '{initiative_name}' is not associated with level '{level_name}'"
            )
            remarktable.loc[idx, rejected_flag_col] = True
            remark = f"{initiative_name} is not associated with this level"
            append_validation_remark_vectorized(
                remarktable, pd.Series([True], index=[idx]), validation_remark_col, remark
            )
            validation_count += 1

    logger.info(
        f"Completed validation check: NPI Initiative Level Association validation - marked {validation_count} records as rejected"
    )
    return remarktable


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    GlobalNPILevels,
    ItemMaster,
    AccountMaster,
    ChannelMaster,
    RegionMaster,
    PnLMaster,
    DemandDomainMaster,
    LocationMaster,
    FileUpload,
    NPIInitiative,
    NPILevel,
    NPIInitiativeLevel,
    NPIInitiativeStatus,
    NPIAssortments,
    DataValidation,
    df_keys,
):
    """NPI Validation Plugin - Validates NPI data and creates initiatives. Returns tuple of output DataFrames."""
    logger.info(f"Executing NPI Validation Plugin for slice {df_keys}")

    cfg = ValidationConfig

    # Master data configuration
    master_dfs_config = _prepare_master_data_config(
        ItemMaster,
        AccountMaster,
        ChannelMaster,
        RegionMaster,
        PnLMaster,
        DemandDomainMaster,
        LocationMaster,
    )

    # Initialize ALL variables at the beginning to avoid NameError
    (
        empty_initiative_dim_data,
        empty_initiative_fact_data,
        empty_initiative_level_fact_data,
        empty_valid_entries,
        empty_initiative_stage_fact_file,
    ) = _initialize_empty_outputs()
    invalid_entries = pd.DataFrame(columns=cfg.invalid_entries_output_req_cols)
    initiative_fact_data = pd.DataFrame(columns=cfg.initiative_fact_data_req_output_cols)
    initiative_level_fact_data = pd.DataFrame(
        columns=cfg.initiative_level_fact_data_req_output_cols
    )
    valid_entries = pd.DataFrame(columns=cfg.valid_entries_output_req_cols)
    initiative_stage_fact_file = pd.DataFrame(
        columns=cfg.initiative_stage_fact_file_req_output_cols
    )

    try:
        # Prepare main validation table
        invalid_entries = _prepare_remarktable(FileUpload)

        if invalid_entries.empty:
            logger.info("No data to process, returning empty outputs with defined columns")
            return (
                invalid_entries,
                initiative_fact_data,
                initiative_level_fact_data,
                valid_entries,
                initiative_stage_fact_file,
            )

        logger.info(f"Starting validation checks for {len(invalid_entries)} records")

        # Run all validation checks using optimized algorithms
        invalid_entries = _run_all_validations(
            invalid_entries,
            master_dfs_config,
            GlobalNPILevels,
            NPILevel,
            NPIInitiative,
            NPIInitiativeLevel,
            NPIInitiativeStatus,
            NPIAssortments,
        )

        # Log validation summary
        total_records = len(invalid_entries)
        rejected_records = invalid_entries[cfg.REJECTED_FLAG_COL].sum()
        approved_records = total_records - rejected_records
        logger.info(
            f"Validation Summary - Total: {total_records}, Approved: {approved_records}, Rejected: {rejected_records}"
        )

        # Create initiatives and generate output tables
        logger.info("Starting initiative creation process")
        (
            initiative_dim_data,
            initiative_fact_data,
            initiative_level_fact_data,
            valid_entries,
            initiative_stage_fact_file,
        ) = _create_all_outputs(
            invalid_entries,
            empty_initiative_dim_data,
            empty_initiative_fact_data,
            empty_initiative_level_fact_data,
            empty_valid_entries,
            empty_initiative_stage_fact_file,
            GlobalNPILevels,
            DataValidation,
        )

        # Clean up and format final validation output
        invalid_entries = _finalize_remarktable_output(invalid_entries)

        # Ensure all output DataFrames have only the defined columns (keeping Validationcheck.py format)
        logger.info("Filtering all output DataFrames to defined columns")
        initiative_fact_data = initiative_fact_data[cfg.initiative_fact_data_req_output_cols]
        initiative_level_fact_data = initiative_level_fact_data[
            cfg.initiative_level_fact_data_req_output_cols
        ]
        valid_entries = valid_entries[cfg.valid_entries_output_req_cols]
        initiative_stage_fact_file = initiative_stage_fact_file[
            cfg.initiative_stage_fact_file_req_output_cols
        ]

        logger.info("Returning all validation and initiative creation outputs")
        return (
            invalid_entries,
            initiative_fact_data,
            initiative_level_fact_data,
            valid_entries,
            initiative_stage_fact_file,
        )

    except Exception as e:
        logger.exception(f"Exception occurred for slice {df_keys}: {e}")
        # Return empty dataframes on error - don't try to process potentially corrupted data
        return (
            pd.DataFrame(columns=cfg.invalid_entries_output_req_cols),  # invalid_entries
            pd.DataFrame(columns=cfg.initiative_fact_data_req_output_cols),  # initiative_fact_data
            pd.DataFrame(
                columns=cfg.initiative_level_fact_data_req_output_cols
            ),  # initiative_level_fact_data
            pd.DataFrame(columns=cfg.valid_entries_output_req_cols),  # valid_entries
            pd.DataFrame(
                columns=cfg.initiative_stage_fact_file_req_output_cols
            ),  # initiative_stage_fact_file
        )
