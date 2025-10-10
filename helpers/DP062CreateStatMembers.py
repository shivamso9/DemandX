import logging
import re

import pandas as pd
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.utils import add_dim_suffix

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None


def get_attribute_value(dim_attribute: str) -> str:
    # Regular expression pattern to match the text within square brackets
    pattern = r"\[(.*?)\]"

    # Using re.search to find the first match of the pattern in the string
    match = re.search(pattern, dim_attribute)

    # If match is found, extract the content within square brackets
    if match:
        result = match.group(1)
        logger.debug(f"result : {result}")
    else:
        result = None

    return result


def get_out_data(
    Dimension: str,
    PlanningGrain: str,
    StatGrain: str,
    MasterData: pd.DataFrame,
    ForecastLevelData: pd.DataFrame,
    StatMemberData: pd.DataFrame,
    version_col: str,
    version: str,
    GroupFilter: pd.DataFrame,
    GROUP_FILTER: str,
    MEMBERS: str,
    GROUP_MEMBERS: str,
    ForecastIterationSelectionData: pd.DataFrame,
    segmentation_lob_col: str = None,
) -> pd.DataFrame:
    # Eg. group_member_col_name = "Stat Item Group Members"
    group_member_col_name = " ".join([get_attribute_value(dim_attribute=StatGrain), GROUP_MEMBERS])

    # extract "Stat Item" from "Item.[Stat Item]"
    attribute = get_attribute_value(dim_attribute=StatGrain)

    # col_name = "Stat Item Members"
    col_name = " ".join([attribute, MEMBERS])

    cols_required_in_out = [
        version_col,
        PlanningGrain,
        col_name,
        group_member_col_name,
    ]
    MasterDataOut = pd.DataFrame(columns=cols_required_in_out)
    try:
        DISPLAY_NAME: str = "Display Name"

        # Eg. relevant_level = "L1"
        relevant_level = ForecastLevelData[f"{Dimension} Level"].iloc[0]

        # if user input is 'All', make it 'All Item'
        relevant_level = add_dim_suffix(input=relevant_level, dim=Dimension)

        logger.debug(f"Dimension : {Dimension}, relevant_level : {relevant_level}")

        # Add "Item Level" column to master data
        MasterData[f"{Dimension} Level"] = ForecastLevelData[f"{Dimension} Level"]

        # Eg. required_col = "Item.[L1]"
        required_col = f"{Dimension}.[" + relevant_level + "]"
        logger.debug(f"required_col : {required_col}")

        # Eg. MembersStat = "Stat Item Members"
        MembersStat = " ".join([get_attribute_value(dim_attribute=StatGrain), MEMBERS])
        logger.debug(f"MembersStat : {MembersStat}")

        # copy data from "Item.[L1]" to "Stat Item Members" column
        MasterData[MembersStat] = MasterData[required_col]

        # copy displayname from "Item.[L1%DisplayName]" to "Stat Item Members Display Name"
        MembersStatDisplayName = MembersStat + " " + DISPLAY_NAME
        required_col_display_name = f"{Dimension}.[" + relevant_level + "$DisplayName" + "]"
        MasterData[MembersStatDisplayName] = MasterData[required_col_display_name]

        # Eg. col_name = "Stat Item Group Filter"
        col_name = " ".join([get_attribute_value(dim_attribute=StatGrain), GROUP_FILTER])

        # Eg. group_filter_attribute = "Item.[L5]"
        group_filter = GroupFilter[col_name].unique()[0]

        # if user input is 'All', make it 'All Item'
        group_filter = add_dim_suffix(input=group_filter, dim=Dimension)

        group_filter_attribute = f"{Dimension}.[" + group_filter + "]"

        # copy data from "Item.[L5]" to "Stat Item Group Members"
        MasterData[group_member_col_name] = MasterData[group_filter_attribute]

        # copy display name data
        group_member_col_name_display_name = group_member_col_name + " " + DISPLAY_NAME
        group_filter_attribute_display_name = f"{Dimension}.[" + group_filter + "$DisplayName" + "]"
        MasterData[group_member_col_name_display_name] = MasterData[
            group_filter_attribute_display_name
        ]

        relevant_fields = [
            PlanningGrain,
            MembersStat,
            MembersStatDisplayName,
            group_member_col_name,
            group_member_col_name_display_name,
        ]

        if Dimension == "Item":

            # seg_lob_mapping_col = "Segmentation LOB Group Filter"
            seg_lob_mapping_col = segmentation_lob_col + " " + GROUP_FILTER

            # seg_lob_col = 'All'
            seg_lob_col = GroupFilter[seg_lob_mapping_col].iloc[0]

            # if user input is 'All', make it 'All Item'
            seg_lob_col_group_filter = add_dim_suffix(input=seg_lob_col, dim=Dimension)

            # seg_lob_col_group_filter_attribute = "Item.[All Item]"
            seg_lob_col_group_filter_attribute = f"{Dimension}.[" + seg_lob_col_group_filter + "]"

            # seg_lob_col_group_filter_attribute_display_name = "Item.[All Item$DisplayName]"
            seg_lob_col_group_filter_attribute_display_name = (
                f"{Dimension}.[" + seg_lob_col_group_filter + "$DisplayName" + "]"
            )

            # create placeholders
            item_segmentation_lob_display_name_col = (
                f"{Dimension}.[" + segmentation_lob_col + "$DisplayName" + "]"
            )
            item_segmentation_lob_col = f"{Dimension}.[" + segmentation_lob_col + "]"

            # copy data frrom "All Item" to Segmentation LOB
            MasterData[item_segmentation_lob_col] = MasterData[seg_lob_col_group_filter_attribute]
            MasterData[item_segmentation_lob_display_name_col] = MasterData[
                seg_lob_col_group_filter_attribute_display_name
            ]

            relevant_fields.extend(
                [
                    item_segmentation_lob_col,
                    item_segmentation_lob_display_name_col,
                ]
            )
        elif Dimension == "Location":
            relevant_fields.extend(
                [
                    "Location.[All Location]",
                    "Location.[All Location$DisplayName]",
                ]
            )

        # retain planning grain and stat grain
        MasterData = MasterData[relevant_fields].drop_duplicates()

        # filter members in planning scope
        planning_members_in_scope = ForecastIterationSelectionData[PlanningGrain].unique().tolist()

        MasterDataOut = MasterData[MasterData[PlanningGrain].isin(planning_members_in_scope)]

        # filter out existing stat members
        # bulk member update query in IBPL which consumes the output of this plugin will fail if it tries to insert existing members
        ExistingStatMembers = StatMemberData[StatGrain].unique().tolist()

        filter_clause = MasterDataOut[MembersStat].isin(ExistingStatMembers)
        MasterDataOut = MasterDataOut[~filter_clause]

        # rename segmentation lob column
        if Dimension == "Item":
            segmentation_lob_members_col = " ".join(
                [
                    segmentation_lob_col,
                    MEMBERS,
                ]
            )
            rename_mapping = {
                item_segmentation_lob_col: segmentation_lob_members_col,
                item_segmentation_lob_display_name_col: segmentation_lob_members_col
                + " "
                + DISPLAY_NAME,
            }
            MasterDataOut.rename(
                columns=rename_mapping,
                inplace=True,
            )
        elif Dimension == "Location":
            MasterDataOut.rename(
                columns={
                    "Location.[All Location]": "All Location Members",
                    "Location.[All Location$DisplayName]": "All Location Members " + DISPLAY_NAME,
                },
                inplace=True,
            )

        if MasterDataOut.empty:
            logger.warning(f"No members to be updated for {Dimension}")

            MasterDataOut = pd.DataFrame(columns=cols_required_in_out)
        else:
            MasterDataOut.insert(loc=0, column=version_col, value=version)
    except Exception as e:
        logger.exception(e)

    return MasterDataOut


col_mapping = {
    "Segmentation LOB Members": "str",
    "Stat Item Group Members": "str",
    "Stat Item Members": "str",
    "Segmentation LOB Members Display Name": "str",
    "Stat Item Group Members Display Name": "str",
    "Stat Item Members Display Name": "str",
    "Stat Region Group Members": str,
    "Stat Region Members": str,
    "Stat Region Group Members Display Name": str,
    "Stat Region Members Display Name": str,
    "Stat Channel Group Members": str,
    "Stat Channel Members": str,
    "Stat Channel Group Members Display Name": str,
    "Stat Channel Members Display Name": str,
    "Stat Account Group Members": str,
    "Stat Account Members": str,
    "Stat Account Group Members Display Name": str,
    "Stat Account Members Display Name": str,
    "All Location Members": str,
    "Stat Location Group Members": str,
    "Stat Location Members": str,
    "All Location Members Display Name": str,
    "Stat Location Group Members Display Name": str,
    "Stat Location Members Display Name": str,
    "Stat Demand Domain Group Members": str,
    "Stat Demand Domain Members": str,
    "Stat Demand Domain Group Members Display Name": str,
    "Stat Demand Domain Members Display Name": str,
    "Stat PnL Group Members": str,
    "Stat PnL Members": str,
    "Stat PnL Group Members Display Name": str,
    "Stat PnL Members Display Name": str,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    ForecastItemLevelData,
    ForecastRegionLevelData,
    ForecastAccountLevelData,
    ForecastChannelLevelData,
    ForecastPnLLevelData,
    ForecastDemandDomainLevelData,
    ForecastLocationLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    StatItemData,
    StatRegionData,
    StatAccountData,
    StatChannelData,
    StatPnLData,
    StatDemandDomainData,
    StatLocationData,
    ForecastIterationSelectionData,
    GroupFilter,
    df_keys,
):
    plugin_name = "DP062CreateStatMembers"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # configurables
    VERSION: str = "Version.[Version Name]"
    FORECAST_ITERATION: str = "Forecast Iteration.[Forecast Iteration]"
    PLANNING_ITEM: str = "Item.[Planning Item]"
    PLANNING_REGION: str = "Region.[Planning Region]"
    PLANNING_CHANNEL: str = "Channel.[Planning Channel]"
    PLANNING_ACCOUNT: str = "Account.[Planning Account]"
    PLANNING_PNL: str = "PnL.[Planning PnL]"
    PLANNING_DEMAND_DOMAIN: str = "Demand Domain.[Planning Demand Domain]"
    SEGMENTATION_LOB: str = "Segmentation LOB"
    LOCATION: str = "Location.[Location]"
    MEMBERS: str = "Members"
    GROUP_MEMBERS: str = " ".join(["Group", MEMBERS])

    DISPLAY_NAME: str = "Display Name"
    STAT_ITEM: str = "Stat Item"
    STAT_ACCOUNT: str = "Stat Account"
    STAT_LOCATION: str = "Stat Location"
    STAT_REGION: str = "Stat Region"
    STAT_DEMAND_DOMAIN: str = "Stat Demand Domain"
    STAT_CHANNEL: str = "Stat Channel"
    STAT_PNL: str = "Stat PnL"

    SEGMENTATION_LOB_MEMBERS: str = " ".join([SEGMENTATION_LOB, MEMBERS])
    STAT_ITEM_GROUP_MEMBERS: str = " ".join([STAT_ITEM, GROUP_MEMBERS])
    STAT_ITEM_MEMBERS: str = " ".join([STAT_ITEM, MEMBERS])

    SEGMENTATION_LOB_MEMBERS_DISPLAY_NAME = " ".join([SEGMENTATION_LOB_MEMBERS, DISPLAY_NAME])
    STAT_ITEM_GROUP_MEMBERS_DISPLAY_NAME = " ".join([STAT_ITEM_GROUP_MEMBERS, DISPLAY_NAME])
    STAT_ITEM_MEMBERS_DISPLAY_NAME = " ".join([STAT_ITEM_MEMBERS, DISPLAY_NAME])

    STAT_REGION_GROUP_MEMBERS: str = " ".join([STAT_REGION, GROUP_MEMBERS])
    STAT_REGION_MEMBERS: str = " ".join([STAT_REGION, MEMBERS])
    STAT_REGION_GROUP_MEMBERS_DISPLAY_NAME = " ".join([STAT_REGION_GROUP_MEMBERS, DISPLAY_NAME])
    STAT_REGION_MEMBERS_DISPLAY_NAME = " ".join([STAT_REGION_MEMBERS, DISPLAY_NAME])

    STAT_CHANNEL_GROUP_MEMBERS: str = " ".join([STAT_CHANNEL, GROUP_MEMBERS])
    STAT_CHANNEL_MEMBERS: str = " ".join([STAT_CHANNEL, MEMBERS])
    STAT_CHANNEL_GROUP_MEMBERS_DISPLAY_NAME = " ".join([STAT_CHANNEL_GROUP_MEMBERS, DISPLAY_NAME])
    STAT_CHANNEL_MEMBERS_DISPLAY_NAME = " ".join([STAT_CHANNEL_MEMBERS, DISPLAY_NAME])

    STAT_ACCOUNT_GROUP_MEMBERS: str = " ".join([STAT_ACCOUNT, GROUP_MEMBERS])
    STAT_ACCOUNT_MEMBERS: str = " ".join([STAT_ACCOUNT, MEMBERS])
    STAT_ACCOUNT_GROUP_MEMBERS_DISPLAY_NAME: str = " ".join(
        [STAT_ACCOUNT_GROUP_MEMBERS, DISPLAY_NAME]
    )
    STAT_ACCOUNT_MEMBERS_DISPLAY_NAME: str = " ".join([STAT_ACCOUNT_MEMBERS, DISPLAY_NAME])

    STAT_PNL_GROUP_MEMBERS: str = " ".join([STAT_PNL, GROUP_MEMBERS])
    STAT_PNL_MEMBERS: str = " ".join([STAT_PNL, MEMBERS])
    STAT_PNL_GROUP_MEMBERS_DISPLAY_NAME = " ".join([STAT_PNL_GROUP_MEMBERS, DISPLAY_NAME])
    STAT_PNL_MEMBERS_DISPLAY_NAME = " ".join([STAT_PNL_MEMBERS, DISPLAY_NAME])

    STAT_DEMAND_DOMAIN_GROUP_MEMBERS: str = " ".join([STAT_DEMAND_DOMAIN, GROUP_MEMBERS])
    STAT_DEMAND_DOMAIN_MEMBERS: str = " ".join([STAT_DEMAND_DOMAIN, MEMBERS])
    STAT_DEMAND_DOMAIN_GROUP_MEMBERS_DISPLAY_NAME = " ".join(
        [STAT_DEMAND_DOMAIN_GROUP_MEMBERS, DISPLAY_NAME]
    )
    STAT_DEMAND_DOMAIN_MEMBERS_DISPLAY_NAME = " ".join([STAT_DEMAND_DOMAIN_MEMBERS, DISPLAY_NAME])

    ALL_LOCATION_MEMBERS = " ".join(["All Location", MEMBERS])
    STAT_LOCATION_GROUP_MEMBERS: str = " ".join([STAT_LOCATION, GROUP_MEMBERS])
    STAT_LOCATION_MEMBERS: str = " ".join([STAT_LOCATION, MEMBERS])
    ALL_LOCATION_MEMBERS_DISPLAY_NAME = " ".join([ALL_LOCATION_MEMBERS, DISPLAY_NAME])
    STAT_LOCATION_GROUP_MEMBERS_DISPLAY_NAME = " ".join([STAT_LOCATION_GROUP_MEMBERS, DISPLAY_NAME])
    STAT_LOCATION_MEMBERS_DISPLAY_NAME = " ".join([STAT_LOCATION_MEMBERS, DISPLAY_NAME])

    GROUP_FILTER: str = "Group Filter"

    ItemOutColumns = [
        VERSION,
        PLANNING_ITEM,
        SEGMENTATION_LOB_MEMBERS,
        STAT_ITEM_GROUP_MEMBERS,
        STAT_ITEM_MEMBERS,
        SEGMENTATION_LOB_MEMBERS_DISPLAY_NAME,
        STAT_ITEM_GROUP_MEMBERS_DISPLAY_NAME,
        STAT_ITEM_MEMBERS_DISPLAY_NAME,
    ]
    ItemMasterDataOut = pd.DataFrame(columns=ItemOutColumns)
    RegionOutColumns = [
        VERSION,
        PLANNING_REGION,
        STAT_REGION_GROUP_MEMBERS,
        STAT_REGION_MEMBERS,
        STAT_REGION_GROUP_MEMBERS_DISPLAY_NAME,
        STAT_REGION_MEMBERS_DISPLAY_NAME,
    ]
    RegionMasterDataOut = pd.DataFrame(columns=RegionOutColumns)
    ChannelOutColumns = [
        VERSION,
        PLANNING_CHANNEL,
        STAT_CHANNEL_GROUP_MEMBERS,
        STAT_CHANNEL_MEMBERS,
        STAT_CHANNEL_GROUP_MEMBERS_DISPLAY_NAME,
        STAT_CHANNEL_MEMBERS_DISPLAY_NAME,
    ]
    ChannelMasterDataOut = pd.DataFrame(columns=ChannelOutColumns)
    AccountOutColumns = [
        VERSION,
        PLANNING_ACCOUNT,
        STAT_ACCOUNT_GROUP_MEMBERS,
        STAT_ACCOUNT_MEMBERS,
        STAT_ACCOUNT_GROUP_MEMBERS_DISPLAY_NAME,
        STAT_ACCOUNT_MEMBERS_DISPLAY_NAME,
    ]
    AccountMasterDataOut = pd.DataFrame(columns=AccountOutColumns)
    PnLOutColumns = [
        VERSION,
        PLANNING_PNL,
        STAT_PNL_GROUP_MEMBERS,
        STAT_PNL_MEMBERS,
        STAT_PNL_GROUP_MEMBERS_DISPLAY_NAME,
        STAT_PNL_MEMBERS_DISPLAY_NAME,
    ]
    PnLMasterDataOut = pd.DataFrame(columns=PnLOutColumns)
    DemandDomainOutColumns = [
        VERSION,
        PLANNING_DEMAND_DOMAIN,
        STAT_DEMAND_DOMAIN_GROUP_MEMBERS,
        STAT_DEMAND_DOMAIN_MEMBERS,
        STAT_DEMAND_DOMAIN_GROUP_MEMBERS_DISPLAY_NAME,
        STAT_DEMAND_DOMAIN_MEMBERS_DISPLAY_NAME,
    ]
    DemandDomainMasterDataOut = pd.DataFrame(columns=DemandDomainOutColumns)
    LocationOutColumns = [
        VERSION,
        LOCATION,
        ALL_LOCATION_MEMBERS,
        STAT_LOCATION_GROUP_MEMBERS,
        STAT_LOCATION_MEMBERS,
        ALL_LOCATION_MEMBERS_DISPLAY_NAME,
        STAT_LOCATION_GROUP_MEMBERS_DISPLAY_NAME,
        STAT_LOCATION_MEMBERS_DISPLAY_NAME,
    ]
    LocationMasterDataOut = pd.DataFrame(columns=LocationOutColumns)

    try:
        if ForecastIterationSelectionData.empty:
            logger.warning("ForecastIterationSelectionData is empty")
            return (
                ItemMasterDataOut,
                RegionMasterDataOut,
                AccountMasterDataOut,
                ChannelMasterDataOut,
                PnLMasterDataOut,
                DemandDomainMasterDataOut,
                LocationMasterDataOut,
            )

        input_version = ForecastIterationSelectionData[VERSION].unique()[0]
        logger.debug(f"input_version : {input_version}")

        assert len(ForecastItemLevelData) > 0, "ForecastItemLevelData is empty ..."
        assert len(ForecastRegionLevelData) > 0, "ForecastRegionLevelData is empty ..."
        assert len(ForecastAccountLevelData) > 0, "ForecastAccountLevelData is empty ..."
        assert len(ForecastChannelLevelData) > 0, "ForecastChannelLevelData is empty ..."
        assert len(ForecastPnLLevelData) > 0, "ForecastPnLLevelData is empty ..."
        assert len(ForecastDemandDomainLevelData) > 0, "ForecastDemandDomainLevelData is empty ..."
        assert len(ForecastLocationLevelData) > 0, "ForecastLocationLevelData is empty ..."

        assert (
            len(ForecastItemLevelData[FORECAST_ITERATION].unique()) == 1
        ), "Data is supplied for more than one iteration ..."

        iteration_value = ForecastItemLevelData[FORECAST_ITERATION].unique()[0]
        logger.debug(f"iteration_value : {iteration_value}")

        ItemMasterDataOut = get_out_data(
            Dimension="Item",
            PlanningGrain=PLANNING_ITEM,
            StatGrain="Item.[Stat Item]",
            MasterData=ItemMasterData,
            ForecastLevelData=ForecastItemLevelData,
            StatMemberData=StatItemData,
            version_col=VERSION,
            version=input_version,
            GroupFilter=GroupFilter,
            GROUP_FILTER=GROUP_FILTER,
            MEMBERS=MEMBERS,
            GROUP_MEMBERS=GROUP_MEMBERS,
            ForecastIterationSelectionData=ForecastIterationSelectionData,
            segmentation_lob_col=SEGMENTATION_LOB,
        )

        RegionMasterDataOut = get_out_data(
            Dimension="Region",
            PlanningGrain=PLANNING_REGION,
            StatGrain="Region.[Stat Region]",
            MasterData=RegionMasterData,
            ForecastLevelData=ForecastRegionLevelData,
            StatMemberData=StatRegionData,
            version_col=VERSION,
            version=input_version,
            GroupFilter=GroupFilter,
            GROUP_FILTER=GROUP_FILTER,
            MEMBERS=MEMBERS,
            GROUP_MEMBERS=GROUP_MEMBERS,
            ForecastIterationSelectionData=ForecastIterationSelectionData,
        )

        ChannelMasterDataOut = get_out_data(
            Dimension="Channel",
            PlanningGrain=PLANNING_CHANNEL,
            StatGrain="Channel.[Stat Channel]",
            MasterData=ChannelMasterData,
            ForecastLevelData=ForecastChannelLevelData,
            StatMemberData=StatChannelData,
            version_col=VERSION,
            version=input_version,
            GroupFilter=GroupFilter,
            GROUP_FILTER=GROUP_FILTER,
            MEMBERS=MEMBERS,
            GROUP_MEMBERS=GROUP_MEMBERS,
            ForecastIterationSelectionData=ForecastIterationSelectionData,
        )

        AccountMasterDataOut = get_out_data(
            Dimension="Account",
            PlanningGrain=PLANNING_ACCOUNT,
            StatGrain="Account.[Stat Account]",
            MasterData=AccountMasterData,
            ForecastLevelData=ForecastAccountLevelData,
            StatMemberData=StatAccountData,
            version_col=VERSION,
            version=input_version,
            GroupFilter=GroupFilter,
            GROUP_FILTER=GROUP_FILTER,
            MEMBERS=MEMBERS,
            GROUP_MEMBERS=GROUP_MEMBERS,
            ForecastIterationSelectionData=ForecastIterationSelectionData,
        )

        PnLMasterDataOut = get_out_data(
            Dimension="PnL",
            PlanningGrain=PLANNING_PNL,
            StatGrain="PnL.[Stat PnL]",
            MasterData=PnLMasterData,
            ForecastLevelData=ForecastPnLLevelData,
            StatMemberData=StatPnLData,
            version_col=VERSION,
            version=input_version,
            GroupFilter=GroupFilter,
            GROUP_FILTER=GROUP_FILTER,
            MEMBERS=MEMBERS,
            GROUP_MEMBERS=GROUP_MEMBERS,
            ForecastIterationSelectionData=ForecastIterationSelectionData,
        )

        DemandDomainMasterDataOut = get_out_data(
            Dimension="Demand Domain",
            PlanningGrain=PLANNING_DEMAND_DOMAIN,
            StatGrain="Demand Domain.[Stat Demand Domain]",
            MasterData=DemandDomainMasterData,
            ForecastLevelData=ForecastDemandDomainLevelData,
            StatMemberData=StatDemandDomainData,
            version_col=VERSION,
            version=input_version,
            GroupFilter=GroupFilter,
            GROUP_FILTER=GROUP_FILTER,
            MEMBERS=MEMBERS,
            GROUP_MEMBERS=GROUP_MEMBERS,
            ForecastIterationSelectionData=ForecastIterationSelectionData,
        )

        LocationMasterDataOut = get_out_data(
            Dimension="Location",
            PlanningGrain=LOCATION,
            StatGrain="Location.[Stat Location]",
            MasterData=LocationMasterData,
            ForecastLevelData=ForecastLocationLevelData,
            StatMemberData=StatLocationData,
            version_col=VERSION,
            version=input_version,
            GroupFilter=GroupFilter,
            GROUP_FILTER=GROUP_FILTER,
            MEMBERS=MEMBERS,
            GROUP_MEMBERS=GROUP_MEMBERS,
            ForecastIterationSelectionData=ForecastIterationSelectionData,
        )

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
    return (
        ItemMasterDataOut,
        RegionMasterDataOut,
        AccountMasterDataOut,
        ChannelMasterDataOut,
        PnLMasterDataOut,
        DemandDomainMasterDataOut,
        LocationMasterDataOut,
    )
