"""
Date utility functions for NPI validation and processing.
"""

import logging
from datetime import datetime

import pandas as pd
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)


def parse_date_flexible(date_str):
    """Parse date string in various formats to datetime with robust validation using dateutil.parser."""
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None

    date_str = str(date_str).strip()

    # Handle common invalid values
    invalid_values = ["nan", "none", "null", "intro date", "intro", "tbd", "pending", "na", "n/a"]
    if date_str.lower() in invalid_values:
        return None

    # Try to extract numeric patterns (like "124" which might be invalid)
    if date_str.isdigit() and len(date_str) <= 4:  # Single numbers like "124" are likely invalid
        return None

    # Use dateutil.parser.parse for flexible date parsing
    try:
        parsed_date = dateutil_parser.parse(date_str, dayfirst=False, yearfirst=True)
        # Validate reasonable date range (1900 to 2100)
        if 1900 <= parsed_date.year <= 2100:
            return parsed_date
        else:
            logger.debug(
                f"Date '{date_str}' parsed but year {parsed_date.year} is outside valid range (1900-2100)"
            )
            return None
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse date '{date_str}' with dateutil.parser: {str(e)}")
        # Fallback to pandas to_datetime for edge cases
        try:
            parsed_date = pd.to_datetime(date_str, errors="coerce")
            if pd.notna(parsed_date):
                # Validate reasonable date range
                if 1900 <= parsed_date.year <= 2100:
                    return parsed_date.to_pydatetime()
                else:
                    logger.debug(
                        f"Date '{date_str}' parsed with pandas but year {parsed_date.year} is outside valid range (1900-2100)"
                    )
                    return None
        except Exception as e:
            logger.debug(f"Failed to parse date '{date_str}' with pandas to_datetime: {str(e)}")
            return None

    return None


def convert_and_validate_date_for_initiative(date_value, date_field_name=""):
    """Convert and validate date for initiative fact data with robust error handling. Returns datetime object or None if invalid."""
    if (
        pd.isna(date_value)
        or str(date_value).strip() == ""
        or str(date_value).strip().lower() == "nan"
    ):
        return None

    # Handle datetime objects directly
    if isinstance(date_value, (datetime, pd.Timestamp)):
        # It's already a datetime object, just normalize the time
        normalized_date = date_value.replace(hour=0, minute=0, second=0, microsecond=0)
        return normalized_date

    # Handle string inputs
    date_str = str(date_value).strip()

    # Parse the date string
    parsed_date = parse_date_flexible(date_str)

    if parsed_date is not None:
        # Set time to 00:00:00
        normalized_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
        return normalized_date
    else:
        logger.warning(f"Could not parse {date_field_name}: '{date_str}' - will be set to None")
        return None


def get_current_date_midnight():
    """Get current date with time set to 00:00:00."""
    current = datetime.now()
    return current.replace(hour=0, minute=0, second=0, microsecond=0)


def format_initiative_date(x):
    """Format date for initiative fact data to MM/DD/YYYY string format."""
    if pd.isna(x) or x is None:
        return ""
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.strftime("%m/%d/%Y")
    else:
        return str(x)


def convert_and_format_date(x, date_field_name=""):
    """Convert and format date to MM/DD/YYYY string format with robust error handling."""
    if pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan":
        return ""

    # Use robust date conversion
    parsed_date = convert_and_validate_date_for_initiative(x, date_field_name)
    if parsed_date is not None:
        return parsed_date.strftime("%m/%d/%Y")
    else:
        return ""  # Return empty string for invalid dates


def format_stage_date_to_mmddyyyy_with_time(x):
    """Format date for stage fact file to MM/DD/YYYY HH:MM:SS string format."""
    if pd.isna(x) or x == "":
        return ""
    try:
        if isinstance(x, str):
            dt = pd.to_datetime(x)
        else:
            dt = x
        return dt.strftime("%m/%d/%Y %H:%M:%S")
    except Exception:
        return str(x)


def is_valid_date(date_str):
    """Check if a date string is valid using parse_date_flexible."""
    return parse_date_flexible(date_str) is not None
