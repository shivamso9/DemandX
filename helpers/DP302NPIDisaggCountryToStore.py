import logging

import pandas as pd
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

# from o9Reference.common_utils.decorators import map_output_columns_to_dtypes

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


@log_inputs_and_outputs
@timed
@convert_category_cols_to_str
def main(Actuals, LikeItem, NPIForecast, Last13Weeks, Active, df_keys):
    # TODO : Change plugin name here
    plugin_name = "DP302NPIDisaggCountryToStore"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    cols_required_in_output1 = [
        "[Version].[Version Name]",
        "[Item].[Planning Item]",
        "[Channel].[Planning Channel]",
        "[Time].[Partial Week]",
        "[Account].[Planning Account]",
        "[Location].[Location Country]",
        "[Region].[Planning Region]",
        "[PnL].[Planning PnL]",
        "[Demand Domain].[Planning Demand Domain]",
        "Stat Fcst Store",
        "TLG Internal Store Fcst",
    ]
    DisaggregationOP = pd.DataFrame(columns=cols_required_in_output1)

    try:
        # TODO : Write all your code here
        if (
            len(Actuals) == 0
            or len(LikeItem) == 0
            or len(NPIForecast) == 0
            or len(Last13Weeks) == 0
        ):
            logger.warning("One of the inputs is Null. Exiting : {} ...".format(df_keys))
            DisaggregationOP = pd.DataFrame(columns=cols_required_in_output1)
            logger.warning("No further execution for this slice")
            return DisaggregationOP

        PWCol = "Time.[Partial Week]"
        Item = "Item.[Planning Item]"
        Location = "Location.[Planning Location]"
        Country = "Location.[Location Country]"
        Channel = "Channel.[Planning Channel]"
        Account = "Account.[Planning Account]"
        Pnl = "PnL.[Planning PnL]"
        DemDom = "Demand Domain.[Planning Demand Domain]"
        Region = "Region.[Planning Region]"
        Version = "Version.[Version Name]"
        LikeItemCol = "Like Item Selected"
        ActualCol = "Actual"
        StatFcst = "Stat Fcst Store"
        IntFcst = "TLG Internal Store Fcst"
        History = "History"
        SumHistory = "SumOfHistory"
        AvgActual = "AvgActual"
        Ratio = "Ratio"
        NPIFcst = "TLG Stat Fcst Adj"
        SFTI = "TLG Internal Country Fcst Adj"

        # Filter Actuals to only contain those items that are present in "Like Item Selected"
        LikeItemsSet = set(LikeItem[LikeItemCol])
        Actuals = Actuals[Actuals[Item].isin(LikeItemsSet)]

        # Create another column in NPIForecast that holds prev years PWs (PrevPW)
        NPIForecast["PrevPW"] = NPIForecast[PWCol].str[:8] + (
            NPIForecast[PWCol].str[8:].astype(int) - 1
        ).astype(str)

        # Merge NPIForecast and Active => to get the active stores in a country, merge NPIForecast and Like Item => To get the PlanningItem and LikeItem in one dataframe
        Active = Active[Active["Active"]]
        Active.drop("Active", axis=1, inplace=True)
        NPIForecast = pd.merge(
            NPIForecast,
            Active,
            on=[Version, Channel, Country, Item],
            how="inner",
        )
        NPIForecast = pd.merge(
            NPIForecast,
            LikeItem,
            on=[Version, Channel, Account, Region, Pnl, DemDom, Country, Item],
            how="left",
        )
        logger.info("Step 1")

        # Merge NPIForecast and Actuals on Like Item and Prev_PWs
        NPIForecast = pd.merge(
            NPIForecast,
            Actuals,
            left_on=[
                Version,
                Channel,
                Account,
                Region,
                Pnl,
                DemDom,
                Location,
                LikeItemCol,
                "PrevPW",
            ],
            right_on=[
                Version,
                Channel,
                Account,
                Region,
                Pnl,
                DemDom,
                Location,
                Item,
                PWCol,
            ],
            how="left",
            suffixes=("", "_actual"),
        )
        NPIForecast.drop(
            ["Item.[Planning Item]_actual", "Time.[Partial Week]_actual"],
            axis=1,
            inplace=True,
        )

        # Filter Actuals to only contain PWs in Last13Weeks, for each item take an average for the duration of Last13Weeks, call it avgActual
        Last13weeksSet = set(Last13Weeks[PWCol])
        Actuals = Actuals[Actuals[PWCol].isin(Last13weeksSet)]
        Actuals[AvgActual] = Actuals.groupby(
            [Version, Channel, Account, Region, Pnl, DemDom, Location, Item]
        )[ActualCol].transform("sum")
        Actuals[AvgActual] = Actuals[AvgActual] / 13.0
        Actuals.drop([PWCol, ActualCol], axis=1, inplace=True)
        Actuals.drop_duplicates(inplace=True)
        Actuals.rename(columns={Item: LikeItemCol}, inplace=True)
        logger.info("Step 2")

        # Merge Actuals and NPIForecast on LikeItem.
        DisaggregationOP = pd.merge(
            NPIForecast,
            Actuals,
            on=[
                Version,
                Channel,
                Account,
                Region,
                Pnl,
                DemDom,
                Location,
                LikeItemCol,
            ],
            how="left",
        )
        DisaggregationOP = DisaggregationOP.fillna(0)

        # History calculation: If Actuals at any PW is zero, put History = avgActual; else put History = 0.4*avgActual + 0.6*Actual
        DisaggregationOP[History] = DisaggregationOP[ActualCol].where(
            DisaggregationOP[ActualCol] == 0,
            0.6 * DisaggregationOP[ActualCol] + 0.4 * DisaggregationOP[AvgActual],
        )
        DisaggregationOP[History] = DisaggregationOP[History].where(
            DisaggregationOP[ActualCol] > 0, DisaggregationOP[AvgActual]
        )

        # For each Country Location, calculate SumHistory for each PW, then ratio: History/SumHistory. If for an Item there's no Like Item or ratio is zero: put ratio as equally divided between countries.
        DisaggregationOP[SumHistory] = DisaggregationOP.groupby(
            [
                Version,
                Channel,
                Account,
                Region,
                Pnl,
                DemDom,
                Country,
                Item,
                PWCol,
            ]
        )[History].transform("sum")
        DisaggregationOP[SumHistory] = DisaggregationOP[SumHistory].replace(0, 1)
        DisaggregationOP[Ratio] = DisaggregationOP[History] / DisaggregationOP[SumHistory]
        logger.info("Step 3")
        """
        def adjust_ratios(group):
            if (group[Ratio] == 0).all():
                group[Ratio] = 1 / len(group)
            return group
        DisaggregationOP = NPIForecast.groupby([Version, Channel, Account, Region, Pnl, DemDom, Location, Item, PWCol]).apply(adjust_ratios).reset_index(drop=True)
        """
        DisaggregationOP["num_locations"] = DisaggregationOP.groupby(
            [
                Version,
                Channel,
                Account,
                Region,
                Pnl,
                DemDom,
                Country,
                Item,
                PWCol,
            ]
        )[Location].transform("count")
        zero_ratios = (
            DisaggregationOP.groupby(
                [
                    Version,
                    Channel,
                    Account,
                    Region,
                    Pnl,
                    DemDom,
                    Country,
                    Item,
                    PWCol,
                ]
            )[Ratio].transform("sum")
            == 0
        )
        DisaggregationOP.loc[zero_ratios, Ratio] = 1 / DisaggregationOP["num_locations"]
        DisaggregationOP.drop(columns="num_locations", inplace=True)

        # Multiply ratio with NPIFcst and StatFcst
        DisaggregationOP[StatFcst] = DisaggregationOP[NPIFcst] * DisaggregationOP[Ratio]
        DisaggregationOP[IntFcst] = DisaggregationOP[SFTI] * DisaggregationOP[Ratio]

        DisaggregationOP.drop(
            [
                Ratio,
                SumHistory,
                History,
                AvgActual,
                ActualCol,
                LikeItemCol,
                "PrevPW",
                NPIFcst,
                SFTI,
                Country,
            ],
            axis=1,
            inplace=True,
        )

        # DisaggregationOP = DisaggregationOP[cols_required_in_output1]

        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        DisaggregationOP = pd.DataFrame(columns=cols_required_in_output1)
    return DisaggregationOP
