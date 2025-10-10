import logging

import pandas as pd
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

logger = logging.getLogger("o9_logger")

col_mapping = {
    "Opportunity Line Stage": str,
    "Opportunity Line Sales Type": str,
    "Opportunity Line Must Win": str,
    "Opportunity Line Probability": float,
    "Opportunity Line Units": float,
    "Opportunity Line Revenue": float,
    "Opportunity Line Created Date": "datetime64[ns]",
    "Opportunity Line Modified Date": "datetime64[ns]",
    "Opportunity Line Requested Date": "datetime64[ns]",
    "Opportunity Line Leasing Duration (in months)": float,
    "Opportunity Line Residual Value": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(Opportunity, df_keys):
    plugin_name = "DP031OpportunityLatestInput"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    cols_required_in_output = [
        "Version.[Version Name]",
        "Opportunity.[Opportunity Line]",
        "Channel.[Channel]",
        "Account.[Account]",
        "PnL.[PnL]",
        "Demand Domain.[Demand Domain]",
        "Region.[Region]",
        "Location.[Planning Location]",
        "Item.[Planning Item]",
        "Opportunity Line Stage",
        "Opportunity Line Sales Type",
        "Opportunity Line Must Win",
        "Opportunity Line Probability",
        "Opportunity Line Units",
        "Opportunity Line Revenue",
        "Opportunity Line Created Date",
        "Opportunity Line Modified Date",
        "Opportunity Line Requested Date",
        "Opportunity Line Leasing Duration (in months)",
        "Opportunity Line Residual Value",
    ]
    output = pd.DataFrame(columns=cols_required_in_output)
    try:

        # checking if data is present or not
        if Opportunity is None:
            logger.info("Opportunity is missing. Exiting the plugin execution")
        else:
            Opportunity["Opportunity Line Must Win Input"].fillna("", inplace=True)
            Opportunity["Time.[Day]"] = pd.to_datetime(Opportunity["Time.[Day]"])
            Opportunity = Opportunity[Opportunity["Opportunity Line Stage Input"] != "Cancelled"]
            output = Opportunity.loc[
                Opportunity["Time.[Day]"] == Opportunity["Time.[Day]"].max()
            ].drop(columns="Time.[Day]", axis=1)
            output.rename(
                columns={
                    "Opportunity Line Stage Input": "Opportunity Line Stage",
                    "Opportunity Line Sales Type Input": "Opportunity Line Sales Type",
                    "Opportunity Line Must Win Input": "Opportunity Line Must Win",
                    "Opportunity Line Probability Input": "Opportunity Line Probability",
                    "Opportunity Line Units Input": "Opportunity Line Units",
                    "Opportunity Line Revenue Input": "Opportunity Line Revenue",
                    "Opportunity Line Created Date Input": "Opportunity Line Created Date",
                    "Opportunity Line Modified Date Input": "Opportunity Line Modified Date",
                    "Opportunity Line Requested Date Input": "Opportunity Line Requested Date",
                    "Opportunity Line Leasing Duration Input (in months)": "Opportunity Line Leasing Duration (in months)",
                    "Opportunity Line Residual Value Input": "Opportunity Line Residual Value",
                },
                inplace=True,
            )
            # filter relevant output columns - sometimes platform adds columns like 'executor_id'
            output = output[cols_required_in_output]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        output = pd.DataFrame(columns=cols_required_in_output)
    return output
