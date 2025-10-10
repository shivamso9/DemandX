import logging

import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants
from helpers.utils import add_dim_suffix, filter_for_iteration

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def remove_count(input: str, dim: str, length: int = 6):
    delimiter = ","
    converted_list = input.split(delimiter)
    converted_list = [x.strip() for x in converted_list]
    length = length * (-1)
    measure = [x[:length] for x in converted_list]
    measure = [(dim + "." + "[" + x + "]") for x in measure]
    return measure


col_mapping = {
    "Planning Item Count": float,
    "L1 Count": float,
    "L2 Count": float,
    "L3 Count": float,
    "L4 Count": float,
    "L5 Count": float,
    "L6 Count": float,
    "Stat Item Count": float,
    "Location Type Count": float,
    "Location Country Count": float,
    "Reporting Location Count": float,
    "Stat Location Count": float,
    "Planning Location Count": float,
    "Location Region Count": float,
    "Planning Account Count": float,
    "Account L1 Count": float,
    "Account L2 Count": float,
    "Account L3 Count": float,
    "Account L4 Count": float,
    "Stat Account Count": float,
    "Planning Demand Domain Count": float,
    "Demand Domain L1 Count": float,
    "Demand Domain L2 Count": float,
    "Demand Domain L3 Count": float,
    "Demand Domain L4 Count": float,
    "Stat Demand Domain Count": float,
    "Planning PnL Count": float,
    "PnL L1 Count": float,
    "PnL L2 Count": float,
    "PnL L3 Count": float,
    "PnL L4 Count": float,
    "Stat PnL Count": float,
    "Channel L1 Count": float,
    "Channel L2 Count": float,
    "Planning Channel Count": float,
    "Stat Channel Count": float,
    "Planning Region Count": float,
    "Region L1 Count": float,
    "Region L2 Count": float,
    "Region L3 Count": float,
    "Region L4 Count": float,
    "Stat Region Count": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    ItemDim,
    LocationDim,
    AccountDim,
    DemandDomainDim,
    PnLDim,
    ChannelDim,
    RegionDim,
    ItemOutputMeasures,
    LocationOutputMeasures,
    AccountOutputMeasures,
    DemandDomainOutputMeasures,
    PnLOutputMeasures,
    ChannelOutputMeasures,
    RegionOutputMeasures,
    StatLevels,
    Version,
    df_keys,
):
    try:
        OutputList = list()
        for the_iteration in StatLevels[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output = decorated_func(
                ItemDim=ItemDim,
                LocationDim=LocationDim,
                AccountDim=AccountDim,
                DemandDomainDim=DemandDomainDim,
                PnLDim=PnLDim,
                ChannelDim=ChannelDim,
                RegionDim=RegionDim,
                ItemOutputMeasures=ItemOutputMeasures,
                LocationOutputMeasures=LocationOutputMeasures,
                AccountOutputMeasures=AccountOutputMeasures,
                DemandDomainOutputMeasures=DemandDomainOutputMeasures,
                PnLOutputMeasures=PnLOutputMeasures,
                ChannelOutputMeasures=ChannelOutputMeasures,
                RegionOutputMeasures=RegionOutputMeasures,
                Version=Version,
                StatLevels=StatLevels,
                df_keys=df_keys,
            )

            OutputList.append(the_output)

        Output = concat_to_dataframe(OutputList)
    except Exception as e:
        logger.exception(e)
        Output = None
    return Output


def processIteration(
    ItemDim,
    LocationDim,
    AccountDim,
    DemandDomainDim,
    PnLDim,
    ChannelDim,
    RegionDim,
    ItemOutputMeasures,
    LocationOutputMeasures,
    AccountOutputMeasures,
    DemandDomainOutputMeasures,
    PnLOutputMeasures,
    ChannelOutputMeasures,
    RegionOutputMeasures,
    Version,
    StatLevels,
    df_keys,
):
    plugin_name = "DP051PopulateDimAttributeCount"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    version_col = "Version.[Version Name]"
    pl_item_count_col = "Planning Item Count"
    location_count_col = "Location Count"
    region_count_col = "Planning Region Count"
    account_count_col = "Planning Account Count"
    demand_domain_count_col = "Demand Domain L1 Count"
    pnl_count_col = "Planning PnL Count"
    channel_count_col = "Channel L1 Count"

    output_cols = [
        version_col,
        pl_item_count_col,
        location_count_col,
        account_count_col,
        demand_domain_count_col,
        pnl_count_col,
        channel_count_col,
        region_count_col,
    ]
    result = pd.DataFrame(columns=output_cols)
    try:
        if StatLevels.empty:
            logger.warning("ForecastGenTimeBucket is empty, cannot process further")
            return result

        StatLevels.drop(o9Constants.VERSION_NAME, axis=1, inplace=True)

        master_data = dict()
        master_data["Item"] = ItemDim
        master_data["Location"] = LocationDim
        master_data["Region"] = RegionDim
        master_data["PnL"] = PnLDim
        master_data["Demand Domain"] = DemandDomainDim
        master_data["Account"] = AccountDim
        master_data["Channel"] = ChannelDim

        for the_column in StatLevels.columns:
            # get the dimension value
            the_dim = the_column.split(" Level")[0]

            # update stat fields respecting the stat level for all dimensions
            the_stat_level = StatLevels[the_column].iloc[0]

            # add dim suffix if member is 'All'
            the_stat_level = add_dim_suffix(input=the_stat_level, dim=the_dim)

            # add dim prefix
            the_stat_level = the_dim + f".[{the_stat_level}]"

            # create the stat column name
            the_stat_col_name = the_dim + f".[Stat {the_dim}]"

            if the_stat_level in master_data[the_dim].columns:
                # update value in dimension dataframe
                master_data[the_dim][the_stat_col_name] = master_data[the_dim][the_stat_level]
            else:
                logger.warning(
                    f"{the_stat_level} column not found in {the_dim} master data, please add this column and run the plugin again ..."
                )

        list_of_output_measures = [
            ItemOutputMeasures,
            LocationOutputMeasures,
            AccountOutputMeasures,
            DemandDomainOutputMeasures,
            PnLOutputMeasures,
            ChannelOutputMeasures,
            RegionOutputMeasures,
        ]
        # take the string in Script Param and extract the attribute out of it
        # in o9 column format
        # eg. "Planning Item Count, L1 Count" would change to "Item.[Planning Item], Item.[L1]"
        location_measure = remove_count(
            input=LocationOutputMeasures,
            dim="Location",
        )
        logger.info("Selected Location attributes are: {}".format(location_measure))
        item_measure = remove_count(
            input=ItemOutputMeasures,
            dim="Item",
        )
        logger.info("Selected Item attributes are: {}".format(item_measure))
        account_measure = remove_count(
            input=AccountOutputMeasures,
            dim="Account",
        )
        logger.info("Selected Account attributes are: {}".format(account_measure))

        demand_domain_measure = remove_count(
            input=DemandDomainOutputMeasures,
            dim="Demand Domain",
        )
        logger.info("Selected Demand Domain attributes are: {}".format(demand_domain_measure))

        pnl_measure = remove_count(
            input=PnLOutputMeasures,
            dim="PnL",
        )
        logger.info("Selected PnL attributes are: {}".format(pnl_measure))

        channel_measure = remove_count(
            input=ChannelOutputMeasures,
            dim="Channel",
        )
        logger.info("Selected Channel attributes are: {}".format(channel_measure))

        region_measure = remove_count(
            input=RegionOutputMeasures,
            dim="Region",
        )
        logger.info("Selected Region attributes are: {}".format(region_measure))

        ItemOutputDim = ItemDim[item_measure]
        LocationOutputDim = LocationDim[location_measure]
        AccountOutputDim = AccountDim[account_measure]
        DemandDomainOutputDim = DemandDomainDim[demand_domain_measure]
        PnLOutputDim = PnLDim[pnl_measure]
        ChannelOutputDim = ChannelDim[channel_measure]
        RegionOutputDim = RegionDim[region_measure]
        list_of_dim_outputs = [
            ItemOutputDim,
            LocationOutputDim,
            AccountOutputDim,
            DemandDomainOutputDim,
            PnLOutputDim,
            ChannelOutputDim,
            RegionOutputDim,
        ]
        result = Version
        for index, outputdim in enumerate(list_of_dim_outputs):
            # Find the number of unique values
            # convert the resulting series to a dataframe
            # take a transpose and drop the Index
            out_df = outputdim.nunique().to_frame().T.reset_index(drop=True)
            # renaming columns
            out_df.columns = list(map(str.strip, list_of_output_measures[index].split(",")))
            result = pd.concat([result, out_df], axis=1)

        # Your code ends here
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        result = pd.DataFrame(columns=output_cols)
    return result
