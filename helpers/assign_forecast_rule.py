"""
Version : 0.0.0
Maintained by : dpref@o9solutions.com
"""

import logging

import pandas as pd

logger = logging.getLogger("o9_logger")


def assign_rules(
    segmentation_output: pd.DataFrame,
    rule_df: pd.DataFrame,
    column_mapping: dict,
    rule_col: str,
    algo_col: str,
    intermittent_col: str,
    plc_col: str,
    los_col: str,
    cov_segment_col: str,
    vol_segment_col: str,
    trend_col: str,
    seasonality_col: str,
) -> pd.DataFrame:
    logger.info("Inside assign_rules function ...")
    try:
        if len(segmentation_output) == 0 or len(rule_df) == 0:
            logger.warning(
                "segmentation_output/rule_df df is empty, cannot assign rules and algorithms ..."
            )
            return segmentation_output

        logger.info(f"segmentation_output, shape : {segmentation_output.shape}")

        # get column mappings
        the_intermittent_col = column_mapping[intermittent_col]
        the_plc_col = column_mapping[plc_col]
        the_los_col = column_mapping[los_col]
        the_cov_segment_col = column_mapping[cov_segment_col]
        the_vol_segment_col = column_mapping[vol_segment_col]
        the_trend_col = column_mapping[trend_col]
        the_seasonality_col = column_mapping[seasonality_col]

        logger.debug("Dropping and resetting index for segmentation_output dataframe ...")

        # reset and drop indices for the boolean filtering to work as expected
        segmentation_output.reset_index(drop=True, inplace=True)

        # sort rules to control evaluation order
        rule_df.sort_values(rule_col, inplace=True)

        # There could be intersections which do not fall into criteria, set up the last rule as fallback
        fall_back_rule_df = rule_df.tail(1)
        fall_back_rule = fall_back_rule_df[rule_col].unique()[0]
        fall_back_algo = fall_back_rule_df[algo_col].unique()[0]

        logger.debug(f"fall_back_rule : {fall_back_rule}")
        logger.debug(f"fall_back_algo : {fall_back_algo}")

        # exclusion criteria
        exclusion_list = ["N/A", "NO MATCH"]

        # exclude the NO MATCH rule from the rule df since this will be used later to fill NAs
        # check if there's a rule which has no match in all conditions
        all_no_match_filter = (
            (rule_df[the_intermittent_col].isin(exclusion_list))
            & (rule_df[the_plc_col].isin(exclusion_list))
            & (rule_df[the_los_col].isin(exclusion_list))
            & (rule_df[the_cov_segment_col].isin(exclusion_list))
            & (rule_df[the_vol_segment_col].isin(exclusion_list))
            & (rule_df[the_trend_col].isin(exclusion_list))
            & (rule_df[the_seasonality_col].isin(exclusion_list))
        )

        all_no_match_rule_df = rule_df[all_no_match_filter]
        if len(all_no_match_rule_df) > 0:

            no_match_rule = all_no_match_rule_df[rule_col].unique()[0]

            # exclude the no match rule from rule df - other wise all rows will satisfy this condition
            # filter clause is constructed using 'and' condition, this requires an all true case which will get satisfied by no match
            rule_df = rule_df[rule_df[rule_col] != no_match_rule]
        segmentation_output[rule_col] = pd.Series([None] * len(segmentation_output))
        # iterate through every row in rule dataframe
        for _, the_rule_row in rule_df.iterrows():

            logger.debug(f"{the_rule_row[rule_col]}\n{the_rule_row}")

            # initialize the filter clause
            the_filter_clause = segmentation_output[rule_col].isna()

            # PLC
            if the_rule_row[the_plc_col].strip() not in exclusion_list:
                plc_clause = segmentation_output[plc_col] == the_rule_row[the_plc_col].strip()
                the_filter_clause = the_filter_clause & plc_clause

            # Intermittency
            if the_rule_row[the_intermittent_col].strip() not in exclusion_list:
                int_clause = (
                    segmentation_output[intermittent_col]
                    == the_rule_row[the_intermittent_col].strip()
                )
                the_filter_clause = the_filter_clause & int_clause

            # Length of Series
            if the_rule_row[the_los_col].strip() not in exclusion_list:
                los_clause = segmentation_output[los_col] == the_rule_row[the_los_col].strip()
                the_filter_clause = the_filter_clause & los_clause

            # COV
            if the_rule_row[the_cov_segment_col].strip() not in exclusion_list:

                # check for in condition - multiple values to match
                if "," in the_rule_row[the_cov_segment_col]:
                    cov_segment_values = the_rule_row[the_cov_segment_col].split(",")
                    cov_segment_values = [x.strip() for x in cov_segment_values]
                    cov_clause = segmentation_output[cov_segment_col].isin(cov_segment_values)
                else:
                    cov_clause = (
                        segmentation_output[cov_segment_col]
                        == the_rule_row[the_cov_segment_col].strip()
                    )

                the_filter_clause = the_filter_clause & cov_clause

            # Volume
            if the_rule_row[the_vol_segment_col].strip() not in exclusion_list:

                # check for in condition - multiple values to match
                if "," in the_rule_row[the_vol_segment_col]:
                    vol_segment_values = the_rule_row[the_vol_segment_col].split(",")
                    vol_segment_values = [x.strip() for x in vol_segment_values]
                    vol_clause = segmentation_output[vol_segment_col].isin(vol_segment_values)
                else:
                    vol_clause = (
                        segmentation_output[vol_segment_col]
                        == the_rule_row[the_vol_segment_col].strip()
                    )

                the_filter_clause = the_filter_clause & vol_clause

            # Trend
            if the_rule_row[the_trend_col].strip() not in exclusion_list:
                trend_clause = segmentation_output[trend_col] == the_rule_row[the_trend_col].strip()
                the_filter_clause = the_filter_clause & trend_clause

            # Seasonality
            if the_rule_row[the_seasonality_col].strip() not in exclusion_list:
                seasonality_clause = (
                    segmentation_output[seasonality_col]
                    == the_rule_row[the_seasonality_col].strip()
                )
                the_filter_clause = the_filter_clause & seasonality_clause

            # apply the filter and update rule, algos
            segmentation_output.loc[the_filter_clause, rule_col] = the_rule_row[rule_col]

            segmentation_output.loc[the_filter_clause, algo_col] = the_rule_row[algo_col]

        logger.debug("Filling nulls using fallback rule and algo ...")
        segmentation_output[rule_col].fillna(fall_back_rule, inplace=True)
        segmentation_output[algo_col].fillna(fall_back_algo, inplace=True)

        logger.debug("Successfully assigned rules ...")
    except Exception as e:
        logger.exception("Exception : {} while assigning forecast rules ...".format(e))

    return segmentation_output
