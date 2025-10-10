import logging

import pandas as pd
from o9Reference.common_utils.decorators import map_output_columns_to_dtypes

logger = logging.getLogger("o9_logger")

col_mapping = {"Include Opportunity (o9 says)": str}


@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    opportunities,
    Parameters,
    df_keys,
):
    plugin_name = "DP033OpportunityShortlist"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Define the variables to capture the column names in the opportunities data frame
    probability_col_name = "Opportunity Probability Final"
    stage_col_name = "Opportunity Stage"
    opportunityType_col_name = "Opportunity Sales Type"
    mustWin_col_name = "Opportunity Must Win"
    include_opp_col_name = "Include Opportunity (o9 says)"

    logger.info("Define the output variables")

    # Get the output column names
    output_columns = [
        "Version.[Version Name]",
        "Opportunity.[Opportunity Line]",
        "Channel.[Channel]",
        "Account.[Account]",
        "PnL.[PnL]",
        "Demand Domain.[Demand Domain]",
        "Region.[Region]",
        "Location.[Planning Location]",
        "Item.[Planning Item]",
        "Time.[Partial Week]",
        include_opp_col_name,
    ]
    output = pd.DataFrame(columns=output_columns)
    try:
        # Extracting the cutoffs from the Parameters dataframe
        probability_param_col_name = "Probability Threshold"
        stage_param_col_name = "List of Opportunity Stages"
        mustWin_param_col_name = "Must Win Options"
        oppType_param_col_name = "List of Opportunity Types"

        logger.info("Started checking for valid input")

        # Check if the required inputs are present or not
        assert len(opportunities) > 0, "No opportunities are present"
        assert len(Parameters) > 0, "No parameters are set"
        assert len(Parameters) == 1, "More than one set of parameters are provided"

        logger.info("Inputs Validation complete")

        logger.info("Include all the opportunities in the forecast")

        # Start by shortlisting all the opportunities
        opportunities[include_opp_col_name] = "Yes"

        logger.info("Reject the opportunities not meeting the probability threshold")

        if ~Parameters[probability_param_col_name].isnull().values.any():
            opportunities.loc[
                opportunities[probability_col_name]
                < Parameters[probability_param_col_name].iloc[0],
                include_opp_col_name,
            ] = "No"

        logger.info("Reject the opportunities not meeting the opportunity stage criteria")

        if ~Parameters[stage_param_col_name].isnull().values.any():
            opportunities.loc[
                ~opportunities[stage_col_name].isin(
                    Parameters[stage_param_col_name].iloc[0].split(",")
                ),
                include_opp_col_name,
            ] = "No"

        logger.info("Reject the opportunities not meeting the Must Win flag criteria")

        opportunities[mustWin_col_name].fillna("", inplace=True)
        opportunities[mustWin_col_name].astype(str)

        Parameters[mustWin_param_col_name].fillna("", inplace=True)

        if ~Parameters[mustWin_param_col_name].isnull().values.any():
            opportunities.loc[
                ~opportunities[mustWin_col_name].isin(
                    Parameters[mustWin_param_col_name].iloc[0].split(",")
                ),
                include_opp_col_name,
            ] = "No"

        logger.info("Reject the opportunities not meeting the opportunity type criteria")

        if ~Parameters[oppType_param_col_name].isnull().values.any():
            opportunities.loc[
                ~opportunities[opportunityType_col_name].isin(
                    Parameters[oppType_param_col_name].iloc[0].split(",")
                ),
                include_opp_col_name,
            ] = "No"

        output = opportunities[output_columns]
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        output = pd.DataFrame(columns=output_columns)
    return output
