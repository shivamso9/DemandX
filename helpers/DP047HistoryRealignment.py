import logging

import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
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


def graph_to_standard(prefix: str, col_list: list) -> list:
    result = []
    prefix = prefix + "."
    for col in col_list:
        col = col.replace(prefix, "", 1)
        col = col.replace("[", "", 1)
        col = col.replace("]", "", 1)
        result.append(col)
    return result


col_mapping = {"Actual Input": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    ActualRaw,
    ItemShipToLocationRealignment,
    ItemLocationRealignment,
    LocationRealignment,
    ShipToRealignment,
    OrderedRealignmentInputList,
    CurrentDay,
    Days,
    df_keys,
):
    plugin_name = "DP047HistoryRealignment"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    version_col = "Version.[Version Name]"
    location_col = "Location.[Location]"
    region_col = "Region.[Region]"
    to_location_col = "to.[Location].[Location]"
    to_time_col = "to.[Time].[Day]"
    timekey_col = "Time.[DayKey]"
    day_col = "Time.[Day]"
    item_col = "Item.[Item]"
    actualraw_col = "Actual Raw"
    actualinput_col = "Actual Input"
    to_shipto_col = "to.[Region].[Region]"
    demand_domain_col = "Demand Domain.[Demand Domain]"
    channel_col = "Channel.[Channel]"
    pnl_col = "PnL.[PnL]"
    account_col = "Account.[Account]"

    cols_required_in_output = [
        version_col,
        item_col,
        location_col,
        region_col,
        demand_domain_col,
        channel_col,
        pnl_col,
        account_col,
        day_col,
        actualinput_col,
    ]

    Output = pd.DataFrame(columns=cols_required_in_output)
    try:
        if len(ActualRaw) == 0:
            logger.warning(
                "No rows found in the DataFrame ActualRaw for the slice : {}...".format(df_keys)
            )
            logger.warning("Will return empty DataFrame for this slice {}...".format(df_keys))
            return Output

        Input_list_values = [
            ItemShipToLocationRealignment,
            ItemLocationRealignment,
            LocationRealignment,
            ShipToRealignment,
        ]
        InputListWithTime = []
        # merging with TimeDimension and filtering for time less than current day
        for i, the_element in enumerate(Input_list_values):
            the_element = pd.merge(
                the_element,
                Days,
                left_on=to_time_col,
                right_on=day_col,
            )
            instance_of_list = the_element
            instance_of_list = instance_of_list[
                instance_of_list[timekey_col] < CurrentDay[timekey_col][0]
            ]
            active_col = instance_of_list.loc[
                :, instance_of_list.columns.str.contains("Active")
            ].columns[0]
            weighted_col = instance_of_list.loc[
                :, instance_of_list.columns.str.contains("Percentage Weighted")
            ].columns[0]
            instance_of_list = instance_of_list[(instance_of_list[active_col])]
            check_nan = instance_of_list[weighted_col].isnull().any()
            if check_nan:
                logger.warning(
                    "For {} the following intersections will be dropped for having nan values in Percentage Weighted Columns".format(
                        OrderedRealignmentInputList.split(",")[i]
                    )
                )
                logger.warning(instance_of_list[instance_of_list[weighted_col].isna()])
            instance_of_list.dropna(inplace=True)
            InputListWithTime.append(instance_of_list)
        list_to_concat = []
        for level, an_element in enumerate(InputListWithTime):
            if len(ActualRaw) == 0:
                break
            entry = an_element
            # take the from only subset
            from_cols_df = entry.loc[:, entry.columns.str.startswith("from")]
            from_cols = from_cols_df.columns.tolist()

            to_cols = [to_shipto_col, to_location_col, to_time_col]
            left_on_cols = graph_to_standard("from", from_cols)
            entry = entry[from_cols]
            entry.drop_duplicates(inplace=True)
            ActualRawmerged = pd.merge(
                ActualRaw,
                entry,
                left_on=left_on_cols,
                right_on=from_cols,
                how="inner",
            )
            ActualRawmerged = ActualRawmerged[ActualRaw.columns]
            ActualRaw = (
                pd.merge(
                    ActualRaw,
                    ActualRawmerged,
                    left_on=[
                        version_col,
                        location_col,
                        region_col,
                        demand_domain_col,
                        channel_col,
                        account_col,
                        pnl_col,
                        item_col,
                        day_col,
                    ],
                    right_on=[
                        version_col,
                        location_col,
                        region_col,
                        demand_domain_col,
                        channel_col,
                        account_col,
                        pnl_col,
                        item_col,
                        day_col,
                    ],
                    how="left",
                    indicator=True,
                )
                .query('_merge == "left_only"')
                .drop("_merge", 1)
            )
            ActualRaw.drop([(actualraw_col + "_y")], axis=1, inplace=True)
            ActualRaw.rename(columns={(actualraw_col + "_x"): actualraw_col}, inplace=True)
            Realignment_instance = an_element
            Realignment_instance.drop([to_time_col, day_col, timekey_col], axis=1, inplace=True)
            ActualRawCartesian = pd.merge(
                ActualRawmerged,
                Realignment_instance,
                how="left",
                left_on=(left_on_cols + [version_col]),
                right_on=(from_cols + [version_col]),
            )
            weighted_col = Realignment_instance.loc[
                :,
                Realignment_instance.columns.str.contains("Percentage Weighted"),
            ].columns[0]
            ActualRawCartesian[actualraw_col] = (
                ActualRawCartesian[actualraw_col] * ActualRawCartesian[weighted_col]
            )
            realignment_cols = list(
                ActualRawCartesian.loc[
                    :, ActualRawCartesian.columns.str.contains("Realignment")
                ].columns
            )
            ActualRawCartesian.drop((from_cols + realignment_cols), axis=1, inplace=True)
            for colname in to_cols:
                if colname in ActualRawCartesian.columns:
                    recolname = colname.replace("to.", "", 1)
                    recolname = recolname.replace("[", "", 1)
                    recolname = recolname.replace("]", "", 1)
                    ActualRawCartesian[recolname] = ActualRawCartesian[colname]
                    ActualRawCartesian.drop(colname, axis=1, inplace=True)
            list_to_concat.append(ActualRawCartesian)

        list_to_concat.append(ActualRaw)
        ActualRawOutput = concat_to_dataframe(list_to_concat)
        ActualRawOutput = pd.merge(
            ActualRawOutput,
            ActualRaw,
            how="outer",
            on=[
                location_col,
                version_col,
                region_col,
                day_col,
                item_col,
                demand_domain_col,
                channel_col,
                pnl_col,
                account_col,
            ],
            indicator=True,
        )
        ActualRawOutput.loc[(ActualRawOutput["_merge"] == "right_only"), (actualraw_col + "_x")] = (
            ActualRawOutput.loc[(ActualRawOutput["_merge"] == "right_only"), (actualraw_col + "_y")]
        )
        ActualRawOutput.drop([(actualraw_col + "_y"), "_merge"], axis=1, inplace=True)
        ActualRawOutput.rename(columns={(actualraw_col + "_x"): actualinput_col}, inplace=True)
        ActualRawOutput[actualinput_col] = ActualRawOutput[actualinput_col]
        ActualRawOutput = ActualRawOutput.groupby(
            [
                version_col,
                item_col,
                location_col,
                region_col,
                demand_domain_col,
                channel_col,
                pnl_col,
                account_col,
                day_col,
            ],
            as_index=False,
        )[actualinput_col].sum()
        Output = ActualRawOutput
        # Your code ends here
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        ActualRawOutput = pd.DataFrame(columns=cols_required_in_output)
    return ActualRawOutput
