import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants
from helpers.utils import filter_for_iteration

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None


def calculate_accuracy(
    df: pd.DataFrame,
    groupby_fields: list,
    lag_measure: str,
    actual_measure: str,
    abs_error_col: str,
    accuracy_col: str,
):
    logger.debug("Calculating accuracy ...")
    all_accuracy = list()
    for the_name, the_group in df.groupby(groupby_fields):
        logger.debug("intersection : {}".format(the_name))
        # calculate abs error
        the_group[abs_error_col] = abs(
            the_group[lag_measure].fillna(0) - the_group[actual_measure].fillna(0)
        )

        # get sum of actuals
        sum_actuals = the_group[actual_measure].sum(skipna=True)
        sum_lag_measure = the_group[lag_measure].sum(skipna=True)

        # forecast not populated but actuals are available
        if np.isnan(sum_lag_measure) and sum_actuals > 0:
            error = 1
        # forecast is populated but actuals are not populated
        elif sum_lag_measure > 0 and np.isnan(sum_actuals):
            error = 1
        # both forecasts and actuals are not populated
        elif np.isnan(sum_lag_measure) and np.isnan(sum_actuals):
            error = 0
        else:
            # calculate sum of abs error, sum of actuals
            sum_abs_error = the_group[abs_error_col].sum(skipna=False)

            # calculate error
            if sum_actuals == 0 and sum_abs_error == 0:
                error = 0
            elif sum_actuals == 0 and sum_abs_error != 0:
                error = 1
            else:
                error = sum_abs_error / sum_actuals

        # calculate accuracy
        if (1 - error) < 0:
            accuracy = 0
        elif (1 - error) > 1:
            accuracy = 1
        else:
            accuracy = 1 - error

        # prepare dataframe
        result = the_group[groupby_fields].drop_duplicates()
        result[accuracy_col] = accuracy
        all_accuracy.append(result)

    return concat_to_dataframe(all_accuracy)


col_mapping = {
    "Stat Fcst Product Accuracy L6P": float,
    "Stat Fcst L1 N6P": float,
    "Stat Fcst L1 N6P LC": float,
    "Stat Fcst L1 Accuracy L6P": float,
    "Stat Fcst Accuracy L6P": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    AccuracyStat,
    StatN6PLC,
    AccuracyProduct,
    AccuracyProductCustomer,
    df_keys,
):
    try:
        AccuracyOutputStatList = list()
        AccuracyOutputProductList = list()
        AccuracyOutputProductCustomerList = list()

        for the_iteration in AccuracyStat[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            (
                the_stat_output,
                the_prod_output,
                the_prod_cust_output,
            ) = decorated_func(
                AccuracyStat=AccuracyStat,
                StatN6PLC=StatN6PLC,
                AccuracyProduct=AccuracyProduct,
                AccuracyProductCustomer=AccuracyProductCustomer,
                df_keys=df_keys,
            )

            AccuracyOutputStatList.append(the_stat_output)
            AccuracyOutputProductList.append(the_prod_output)
            AccuracyOutputProductCustomerList.append(the_prod_cust_output)

        AccuracyOutputStat = concat_to_dataframe(AccuracyOutputStatList)
        AccuracyOutputProduct = concat_to_dataframe(AccuracyOutputProductList)
        AccuracyOutputProductCustomer = concat_to_dataframe(AccuracyOutputProductCustomerList)

    except Exception as e:
        logger.exception(e)
        (
            AccuracyOutputStat,
            AccuracyOutputProduct,
            AccuracyOutputProductCustomer,
        ) = (None, None, None)
    return (
        AccuracyOutputStat,
        AccuracyOutputProduct,
        AccuracyOutputProductCustomer,
    )


def processIteration(
    AccuracyStat,
    StatN6PLC,
    AccuracyProduct,
    AccuracyProductCustomer,
    df_keys,
):
    plugin_name = "DP017L6PAccuracy"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    lag_measure = "Stat Fcst Lag1"
    actual_measure = "Actual"
    abs_error_col = "Absolute Error"
    version_col = "Version.[Version Name]"
    pl_item_col = "Item.[Planning Item]"
    planning_channel_col = "Channel.[Planning Channel]"
    planning_account_col = "Account.[Planning Account]"
    planning_pnl_col = "PnL.[Planning PnL]"
    planning_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    planning_region_col = "Region.[Planning Region]"

    input_version = AccuracyStat[version_col].unique()[0]
    stat_fcst_lc_col = "Stat Fcst LC"
    stat_fcst_l1_col = "Stat Fcst L1"

    # output measures
    accuracy_col = "Stat Fcst L1 Accuracy L6P"
    pl_item_cust_group_accuracy_col = "Stat Fcst Accuracy L6P"
    pl_item_accuracy_col = "Stat Fcst Product Accuracy L6P"
    stat_fcst_l1_output_col = "Stat Fcst L1 N6P"
    stat_fcst_lc_output_col = "Stat Fcst L1 N6P LC"

    stat_groupby_fields = [
        "Item.[Stat Item]",
        "Account.[Stat Account]",
        "Channel.[Stat Channel]",
        "Region.[Stat Region]",
        "PnL.[Stat PnL]",
        "Demand Domain.[Stat Demand Domain]",
        "Location.[Stat Location]",
    ]
    try:
        logger.info("Calculating accuracy at {} ...".format(stat_groupby_fields))

        # calculate stat accuracy
        AccuracyOutputStat = calculate_accuracy(
            df=AccuracyStat,
            groupby_fields=stat_groupby_fields,
            lag_measure=lag_measure,
            actual_measure=actual_measure,
            abs_error_col=abs_error_col,
            accuracy_col=accuracy_col,
        )

        AccuracyOutputStat.insert(0, version_col, input_version)

        logger.info("------ AccuracyOutputStat -------")
        logger.info(AccuracyOutputStat)

        N6PLCFcst = StatN6PLC.groupby(stat_groupby_fields).sum()[[stat_fcst_lc_col]].reset_index()
        N6PLCFcst.rename(columns={stat_fcst_lc_col: stat_fcst_lc_output_col}, inplace=True)

        N6PFcst = StatN6PLC.groupby(stat_groupby_fields).sum()[[stat_fcst_l1_col]].reset_index()
        N6PFcst.rename(columns={stat_fcst_l1_col: stat_fcst_l1_output_col}, inplace=True)

        # Find last 6 month accuracy at Planning Item x Customer Group
        to_groupby = [
            pl_item_col,
            planning_channel_col,
            planning_account_col,
            planning_pnl_col,
            planning_demand_domain_col,
            planning_region_col,
        ]
        logger.info("Calculating accuracy at {} ...".format(to_groupby))
        AccuracyOutputProductCustomer = calculate_accuracy(
            df=AccuracyProductCustomer,
            groupby_fields=to_groupby,
            lag_measure=lag_measure,
            actual_measure=actual_measure,
            abs_error_col=abs_error_col,
            accuracy_col=pl_item_cust_group_accuracy_col,
        )
        AccuracyOutputProductCustomer.insert(0, version_col, input_version)

        # Find last 6 month accuracy at Planning Item
        to_groupby = [pl_item_col]
        logger.info("Calculating accuracy at {} ...".format(to_groupby))
        AccuracyOutputProduct = calculate_accuracy(
            df=AccuracyProduct,
            groupby_fields=to_groupby,
            lag_measure=lag_measure,
            actual_measure=actual_measure,
            abs_error_col=abs_error_col,
            accuracy_col=pl_item_accuracy_col,
        )
        AccuracyOutputProduct.insert(0, version_col, input_version)

        logger.info("Combining N6PLCFcst with N6PFcst ...")
        N6PFcstLCOutput = N6PLCFcst.merge(N6PFcst, on=stat_groupby_fields, how="inner")
        N6PFcstLCOutput.insert(0, version_col, input_version)

        logger.info("Combining N6PFcstLCOutput with AccuracyStat ...")

        AccuracyOutputStat = AccuracyOutputStat.merge(
            N6PFcstLCOutput,
            on=[version_col] + stat_groupby_fields,
            how="outer",
        )

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )

        AccuracyOutputStat_cols = (
            [version_col]
            + stat_groupby_fields
            + [accuracy_col, stat_fcst_l1_output_col, stat_fcst_lc_output_col]
        )
        AccuracyOutputStat = pd.DataFrame(columns=AccuracyOutputStat_cols)

        AccuracyOutputProduct_cols = [
            version_col,
            pl_item_col,
            pl_item_accuracy_col,
        ]
        AccuracyOutputProduct = pd.DataFrame(columns=AccuracyOutputProduct_cols)

        AccuracyOutputProductCustomer_cols = [
            version_col,
            planning_channel_col,
            planning_account_col,
            planning_pnl_col,
            planning_demand_domain_col,
            planning_region_col,
            pl_item_col,
            pl_item_cust_group_accuracy_col,
        ]
        AccuracyOutputProductCustomer = pd.DataFrame(columns=AccuracyOutputProductCustomer_cols)

    return (
        AccuracyOutputStat,
        AccuracyOutputProduct,
        AccuracyOutputProductCustomer,
    )
