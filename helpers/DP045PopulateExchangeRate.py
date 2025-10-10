import logging

import pandas as pd
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


col_mapping = {
    "Exchange Rate": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    CurrencyExchangeRates,
    IsSingleCurrency,
    DefaultInputCurrency,
    CustomerGroupCurrency,
    df_keys,
):
    plugin_name = "DP045PopulateExchangeRate"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    IS_SINGLE_CURRENCY = "Is Single Currency"
    DEFAULT_INPUT_CURRENCY = "Default Input Currency"
    VERSION = "Version.[Version Name]"
    FROM_CURRENCY = "from.[Currency].[Currency]"
    TO_CURRENCY = "to.[Currency].[Currency]"
    TO_MONTH = "to.[Time].[Month]"
    EXCHANGE_RATE_INPUT = "400 Exchange Rate Input.[Exchange Rate Input]"
    LOCAL_CURRENCY = "Region.[Local Currency]"
    EXCHANGE_RATE = "Exchange Rate"
    PLANNING_REGION = "Region.[Planning Region]"
    CURRENCY = "Currency.[Currency]"
    MONTH = "Time.[Month]"
    key = "key"
    cusgrpcurr = "CustomerGroupCurrency"
    defincurr = "DefaultInputCurrency"
    curexrate = "CurrencyExchangeRates"

    # single currency == 1
    cols_required_in_output = [
        VERSION,
        CURRENCY,
        MONTH,
        EXCHANGE_RATE,
    ]
    CurrencyOutput = pd.DataFrame(columns=cols_required_in_output)
    try:
        if IsSingleCurrency[IS_SINGLE_CURRENCY][0] == 0 and len(CustomerGroupCurrency) == 0:
            logger.warning(
                "No Data found in {}, for slice {}. \n Returning Empty Dataframe".format(
                    cusgrpcurr, df_keys
                )
            )
            return CurrencyOutput
        if IsSingleCurrency[IS_SINGLE_CURRENCY][0] == 1 and len(DefaultInputCurrency) == 0:
            logger.warning(
                "No Data found in {}, for slice {}. \n Returning empty Dataframe".format(
                    defincurr, df_keys
                )
            )
            return CurrencyOutput
        if len(CurrencyExchangeRates) == 0:
            logger.warning(
                "No Data found in {}, for slice {}. \n Returning empty Dataframe".format(
                    curexrate, df_keys
                )
            )
            return CurrencyOutput

        # CASE I: SINGLE CURRENCY IS 1
        if IsSingleCurrency[IS_SINGLE_CURRENCY][0] == 1:

            logger.info("Executing Case I: Is Single Currency = 1")
            base_currdf = DefaultInputCurrency[DEFAULT_INPUT_CURRENCY][0]
            filtereddf = CurrencyExchangeRates[
                (CurrencyExchangeRates[FROM_CURRENCY] == base_currdf)
            ]
            # check if the DefaultInputCurrency exists in CurrencyExchangeRates
            if len(filtereddf) == 0:
                logger.warning(
                    "{} does not exist in {}. \n Returning empty Dataframe".format(
                        base_currdf, curexrate
                    )
                )
                return CurrencyOutput

            # keeping only necessary columns in filtereddf and exchange_currlist
            exchange_currlist = filtereddf[[VERSION, TO_CURRENCY, TO_MONTH, EXCHANGE_RATE_INPUT]]
            filtereddf = filtereddf[[VERSION, FROM_CURRENCY, TO_MONTH, EXCHANGE_RATE_INPUT]]
            # renaming columns accordingly
            filtereddf.columns = cols_required_in_output
            # exchange rate of a currency to itself would be 1
            filtereddf[EXCHANGE_RATE] = 1
            filtereddf.drop_duplicates(subset=MONTH, keep="first", inplace=True)
            exchange_currlist.columns = cols_required_in_output
            CurrencyOutput = pd.concat([exchange_currlist, filtereddf], axis=0)
        else:
            logger.info("Executing Case II: Is Single Currency = 0")

            cols_required_in_output = [
                VERSION,
                CURRENCY,
                PLANNING_REGION,
                MONTH,
                EXCHANGE_RATE,
            ]

            base_currdf = CurrencyExchangeRates[
                [
                    VERSION,
                    FROM_CURRENCY,
                    TO_CURRENCY,
                    TO_MONTH,
                    EXCHANGE_RATE_INPUT,
                ]
            ]
            base_currdf.columns = [
                VERSION,
                FROM_CURRENCY,
                CURRENCY,
                MONTH,
                EXCHANGE_RATE_INPUT,
            ]
            base_currdf[key] = 1
            CustomerGroupCurrency[key] = 1
            # creating cartesian product using merge statement and leaving join type to default = 'inner'
            cartproduct = pd.merge(base_currdf, CustomerGroupCurrency, on=key)
            cartproduct.rename(columns={EXCHANGE_RATE_INPUT: EXCHANGE_RATE}, inplace=True)
            cartproduct.loc[
                cartproduct[CURRENCY] == cartproduct[LOCAL_CURRENCY],
                EXCHANGE_RATE,
            ] = 1
            cartproduct = cartproduct[
                (cartproduct[FROM_CURRENCY] == cartproduct[LOCAL_CURRENCY])
                | (cartproduct[LOCAL_CURRENCY] == cartproduct[CURRENCY])
            ]
            cartproduct.drop([key, LOCAL_CURRENCY, FROM_CURRENCY], axis=1, inplace=True)
            CurrencyOutput = cartproduct[cols_required_in_output]

        CurrencyOutput = CurrencyOutput[cols_required_in_output]
        logger.info(f"Plugin {plugin_name} completed ...")
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
    return CurrencyOutput
