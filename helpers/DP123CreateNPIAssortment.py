"""Create Assortment Plugin for Flexible NPI.

Pseudocode:
    Version: 2025.08.00
    --------------------
        - Calculate NPI Association L0
            - Case 1: If Similar Item Scope available
                - Aggregate Assortment Final at User specified level.
                - User specified level is the 'InitiativeLevel' input level for the user provided 'Level' input.
                - Assortment intersection eligible for the 'SimilarItemScope'. Filter it using SimilarItemScope.
                - Cross product the NPI Items ('NPIItemScope') with the assortment intersection to create the active intersection for the NPI items.
                - Rename the columns to NPI Levels.
            - Case 2: If Similar Item Scope does not available
                - Cross product of user specified NPI attributes scope.
            - Case 3: If AssortmentType is FileUpload then NPI Association L0 will be NPIAssortmentFileUpload.
                - Disaggregate the NPI Association L0 to the lower level for the NPI Assortment Final by Level calculation.
        - Calculate NPI Assortment Final by Level
            - Case 1: If Similar Item Scope available
                - Disaggregated the NPI Association L0 to the lower level.
            - Case 2: If Similar Item Scope does not available
                - Disaggregate NPI Association L0
                - Check if disaggregated intersection available in AssortmentFinal input
                - If availabel it will be the output for the intersection.
                - If not present then drop one column using 'AssortmentSequence' and search the intersection in AssormentFinal.
                - If not present repeat above step.
                - If data found, apply cross product between the data and dropped columns.
        - Calculate NPI Assortment Final
            - Null out 'NPIAssortmentFinal' input.
            - Concat 'NPIAssortmentFinalByLevel' input with NPI Assortment Final by Level.
            - Select the high sequence Level from InitiativeLevel input
            - Filter the concat data with the selected Level. If no data found in concat data then select another lower sequence Level and so on, until get the caoncat data.
            - Make the assortment active for the filtered concat data.
        - Calculate NPI Planning Assortment by Level
            - Aggregate 'NPI Assortment Final by Level' at planning level.
            - Group the aggregated data based on 'NPI Association L0'.
            - Equally distribute percentage for each 'NPI Association L0' intersection in the aggregated data.
"""

import logging
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from o9Reference.common_utils.dataframe_utils import (
    column_special_char_remover,
    column_special_char_remover_bulk,
    concat_to_dataframe,
    concat_to_polar_dataframe,
    create_cartesian_product,
    create_cartesian_product_polar,
)
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logger = logging.getLogger("o9_logger")


def user_selected_level(
    InitiativeLevel_required,
    npi_item_level_col,
    npi_account_level_col,
    npi_channel_level_col,
    npi_region_level_col,
    npi_pnl_level_col,
    npi_dd_level_col,
    npi_location_level_col,
):
    """Return formatted columns of user selected level."""
    # Selecting the required columns
    InitiativeLevel_required = InitiativeLevel_required[
        [
            npi_item_level_col,
            npi_account_level_col,
            npi_channel_level_col,
            npi_region_level_col,
            npi_pnl_level_col,
            npi_dd_level_col,
            npi_location_level_col,
        ]
    ].drop_duplicates()

    # Formatting the cols name
    for col in InitiativeLevel_required.columns:
        if "NPI" in col and "Level" in col:
            attribute = col.split("NPI")[1].split("Level")[0].strip()
            InitiativeLevel_required[col] = (
                str(attribute) + ".[" + InitiativeLevel_required[col] + "]"
            )

    return list(InitiativeLevel_required.iloc[0])


def merge_two(df1, df2_key):
    """Merge two dfs."""
    key, df2 = df2_key
    df2 = df2.loc[:, ~df2.columns.duplicated()]
    # if user levels are same, that means, the req column is already present
    if len(df2.columns) == 1:
        return df1
    return pd.merge(df1, df2, on=key, how="left")


def get_level_contains_data(NPIAssortmentFinalByLevel_combined, InitiativeLevel, data_object_col):
    """Get the latest level that contains the data."""
    levels_in_combined_data = NPIAssortmentFinalByLevel_combined[data_object_col].unique()
    for req_level in InitiativeLevel[data_object_col].unique():
        if req_level in levels_in_combined_data:
            return req_level
    return None


def get_lower_level_ass_data(
    NPIAssortmentFinalByLevel, Initiative, Level, InitiativeLevel, initiative_col, data_object_col
):
    """Get lower level assorment data."""
    level_seq = list(InitiativeLevel[data_object_col].unique())
    index = level_seq.index(Level)
    level_seq = level_seq[:index]
    if len(level_seq) == 0:
        raise ValueError("Lower level data does not found")
    level_seq_relevant = level_seq[-1]
    the_NPIAssortmentFinalByLevel = NPIAssortmentFinalByLevel[
        NPIAssortmentFinalByLevel[initiative_col] == Initiative
    ]
    the_NPIAssortmentFinalByLevel = the_NPIAssortmentFinalByLevel[
        the_NPIAssortmentFinalByLevel[data_object_col] == level_seq_relevant
    ]
    return the_NPIAssortmentFinalByLevel


def split_to_list(str, Delimiter):
    """Validate & Split the string to list by Delimiter."""
    if Delimiter not in str:
        logger.warning(
            f"Provided Delimiter: {Delimiter} is not available in the string: {str}. Please check the input"
        )
        logger.info("Setting the Default Delimiter to ,")
        Delimiter = ","

    str = str.split(Delimiter)
    str = [x.strip() for x in str if x not in [None, "na", "NA", "None", "", " ", np.nan]]
    return str


col_mapping = {
    "NPI Association L0": float,
    "NPI Assortment Final by Level": float,
    "NPI Assortment Final": float,
    "NPI Planning Assortment by Level": float,
    "NPI System Split %": float,
}


def find_intersections_using_polar(
    # User col names info
    user_item_level_col,
    user_account_level_col,
    user_channel_level_col,
    user_region_level_col,
    user_pnl_level_col,
    user_dd_level_col,
    user_loc_level_col,
    AssortmentSequence,
    # DataFrames
    AssortmentFinal,
    grainular_level_assortments,
    # Others
    df_keys,
) -> pd.DataFrame:
    """Create the intersections using polar."""
    logger.info("Converting pandas dataframe to polar dataframe ...")
    # Converting to polar dataframes
    grainular_level_assortments = pl.from_pandas(grainular_level_assortments)
    AssortmentFinal_pl = pl.from_pandas(AssortmentFinal)

    logger.info("Converting column name of dataframes to polar compatibility ...")
    # Renaming columns to remove special characters
    AssortmentFinal_cols_pl = column_special_char_remover_bulk(AssortmentFinal_pl.columns)
    AssortmentFinal_pl = AssortmentFinal_pl.rename(AssortmentFinal_cols_pl)

    grainular_level_assortments_cols_pl = column_special_char_remover_bulk(
        grainular_level_assortments.columns
    )
    grainular_level_assortments = grainular_level_assortments.rename(
        grainular_level_assortments_cols_pl
    )

    user_item_level_col_pl = column_special_char_remover(user_item_level_col)
    user_account_level_col_pl = column_special_char_remover(user_account_level_col)
    user_channel_level_col_pl = column_special_char_remover(user_channel_level_col)
    user_region_level_col_pl = column_special_char_remover(user_region_level_col)
    user_pnl_level_col_pl = column_special_char_remover(user_pnl_level_col)
    user_dd_level_col_pl = column_special_char_remover(user_dd_level_col)
    user_loc_level_col_pl = column_special_char_remover(user_loc_level_col)

    Groups = grainular_level_assortments.group_by(
        [
            user_item_level_col_pl,
            user_account_level_col_pl,
            user_channel_level_col_pl,
            user_region_level_col_pl,
            user_pnl_level_col_pl,
            user_dd_level_col_pl,
            user_loc_level_col_pl,
        ]
    )

    assormentfinal_key = AssortmentSequence
    assormentfinal_key_map_pl = column_special_char_remover_bulk(assormentfinal_key)
    assormentfinal_key_pl = list(assormentfinal_key_map_pl.values())

    npi_assortment_final_by_level_df_list = []

    i = 0
    GroupHeight = Groups.agg([]).height

    logger.info("Creating intersections.")
    logger.info(f"Total loops: {GroupHeight}")
    for key, the_data in Groups:

        # print the progress logs
        if i % 100 == 0:
            logger.info(f"{i} Executing out of {GroupHeight}")
        i = i + 1

        data = the_data[assormentfinal_key_pl].unique()
        assortmentfinal_key_rolling_pl = assormentfinal_key_pl.copy()

        for _ in assormentfinal_key_pl:
            # Checking the data availability
            the_data_with_assortmentfinal_pl = (
                data[assortmentfinal_key_rolling_pl]
                .unique()
                .join(
                    AssortmentFinal_pl[assortmentfinal_key_rolling_pl].unique(),
                    left_on=assortmentfinal_key_rolling_pl,
                    right_on=assortmentfinal_key_rolling_pl,
                    how="inner",
                )
            )

            # Check if iteration is last one and no data found
            last_iteration_no_data_condition = (
                len(the_data_with_assortmentfinal_pl) == 0
                and len(assortmentfinal_key_rolling_pl)
                == 1  # Case if no data found and iteration reach to the end, need to create cartesian between each attributes.
            )

            if (
                len(the_data_with_assortmentfinal_pl) == 0
                and len(assortmentfinal_key_rolling_pl) > 1
            ):
                # No data found
                assortmentfinal_key_rolling_pl.pop(0)

            elif (
                len(the_data_with_assortmentfinal_pl)
                != 0  # Case if found data, apply cartesian with dropped attributes.
                or last_iteration_no_data_condition is True
            ):

                # If last iteration and no data found then cartesian with all the attributes hence drop the last attribute
                if last_iteration_no_data_condition is True:
                    assortmentfinal_key_rolling_pl.pop(0)

                # Found data, create cross product with dropped columns or if does not found any data need to apply cartesian on all the attributes.
                dropped_columns = list(
                    set(assormentfinal_key_pl) - set(assortmentfinal_key_rolling_pl)
                )

                if len(dropped_columns) != 0:
                    # If columns dropped, Cartesian it with the found data

                    dropped_columns_data_pl = the_data[dropped_columns].unique()

                    # Create cartesian:
                    # 1.Cartesian between dropped columns
                    # 2.Cartesian between dropped columns and founded data

                    # Create cartesian between the dropped cols
                    for col in dropped_columns_data_pl.columns:
                        # break
                        if dropped_columns_data_pl.columns[0] == col:
                            dropped_columns_data_cartesian_pl = dropped_columns_data_pl[
                                [col]
                            ].unique()
                        else:
                            dropped_columns_data_cartesian_pl = create_cartesian_product_polar(
                                dropped_columns_data_cartesian_pl,
                                dropped_columns_data_pl[[col]].unique(),
                            )

                    # Create cartesian between dropped columns and founded data.
                    if (
                        len(
                            the_data_with_assortmentfinal_pl.select(
                                assortmentfinal_key_rolling_pl
                            ).columns
                        )
                        != 0
                    ):
                        npi_assortment_final_by_level_pl = create_cartesian_product_polar(
                            dropped_columns_data_cartesian_pl,
                            the_data_with_assortmentfinal_pl[
                                assortmentfinal_key_rolling_pl
                            ].unique(),
                        )
                    else:
                        npi_assortment_final_by_level_pl = dropped_columns_data_cartesian_pl

                    npi_assortment_final_by_level_df_list.append(npi_assortment_final_by_level_pl)

                    # Got the data, no need to drop any further columns
                    break

                else:
                    # found the data without dropping any columns
                    npi_assortment_final_by_level_df_list.append(the_data_with_assortmentfinal_pl)
                    break

            elif (
                len(the_data_with_assortmentfinal_pl) == 0
                and len(assortmentfinal_key_rolling_pl) == 0
            ):
                # Could not found any data.
                logger.warning(
                    "Could not found assortment data for NPI Assortment Final by Level for slice {} for NPI Association L0 {}...".format(
                        df_keys, key
                    )
                )

    # Assortment at lower level for case 2 (if Similar Item is missing)
    npi_assortment_final_by_level_output_pl = concat_to_polar_dataframe(
        npi_assortment_final_by_level_df_list
    )

    npi_assortment_final_by_level_output_pl = npi_assortment_final_by_level_output_pl[
        assormentfinal_key_pl
    ]

    logger.info("Converting polar dataframe to pandas dataframe")
    # Converting back to pandas DataFrame
    npi_assortment_final_by_level_output = npi_assortment_final_by_level_output_pl.to_pandas()

    logger.info("Remapping the original column name")
    # Realign the col names
    assormentfinal_key_map_pl_reversed = {v: k for k, v in assormentfinal_key_map_pl.items()}
    npi_assortment_final_by_level_output.rename(
        columns=assormentfinal_key_map_pl_reversed, inplace=True
    )

    logger.info("Successfully created the intersections using polar")
    return npi_assortment_final_by_level_output


def single_initiative_and_level(
    # Params
    Initiative: Optional[str] = None,
    Level: Optional[str] = None,
    AssortmentType: Optional[str] = None,
    AssortmentSequence: Optional[str] = None,
    # NPI Scope
    NPIItemScope: Optional[str] = None,
    NPIAccountScope: Optional[str] = None,
    NPIChannelScope: Optional[str] = None,
    NPIRegionScope: Optional[str] = None,
    NPIPnLScope: Optional[str] = None,
    NPIDemandDomainScope: Optional[str] = None,
    NPILocationScope: Optional[str] = None,
    # Data
    AssortmentFinal: pd.DataFrame = None,
    NPIAssortmentFileUpload: pd.DataFrame = None,
    InitiativeLevel: pd.DataFrame = None,
    NPIAssortmentFinal: pd.DataFrame = None,
    NPIAssortmentFinalByLevel: pd.DataFrame = None,
    # Master data
    ItemMaster: pd.DataFrame = None,
    AccountMaster: pd.DataFrame = None,
    ChannelMaster: pd.DataFrame = None,
    RegionMaster: pd.DataFrame = None,
    PnLMaster: pd.DataFrame = None,
    DemandDomainMaster: pd.DataFrame = None,
    LocationMaster: pd.DataFrame = None,
    # Others
    Delimiter: str = ",",
    ReadFromHive: str = "False",
    df_keys: Optional[dict] = None,
):
    """Entry point of the script for a single initiative and level."""
    plugin_name = "DP123CreateNPIAssortment"
    logger.info(
        "Executing {} for Initiative: {} Level: {} slice: {}".format(
            plugin_name, Initiative, Level, df_keys
        )
    )

    # Configurables
    version_col = "Version.[Version Name]"
    pl_item_col = "Item.[Planning Item]"
    item_col = "Item.[Item]"
    loc_col = "Location.[Location]"
    initiative_col = "Initiative.[Initiative]"
    data_object_col = "Data Object.[Data Object]"
    planning_channel_col = "Channel.[Planning Channel]"
    planning_account_col = "Account.[Planning Account]"
    planning_pnl_col = "PnL.[Planning PnL]"
    planning_demand_domain_col = "Demand Domain.[Planning Demand Domain]"
    planning_region_col = "Region.[Planning Region]"
    planning_location_col = "Location.[Planning Location]"

    npi_item_col = "Item.[NPI Item]"
    npi_account_col = "Account.[NPI Account]"
    npi_channel_col = "Channel.[NPI Channel]"
    npi_region_col = "Region.[NPI Region]"
    npi_pnl_col = "PnL.[NPI PnL]"
    npi_location_col = "Location.[NPI Location]"
    npi_demand_domain_col = "Demand Domain.[NPI Demand Domain]"
    email_col = "Personnel.[Email]"

    npi_association_l0_col = "NPI Association L0"
    npi_assorment_final_by_level_col = "NPI Assortment Final by Level"
    npi_assortment_final_col = "NPI Assortment Final"
    pl_level_assortmnet_col = "NPI Planning Assortment by Level"
    sys_split_perc_col = "NPI System Split %"
    npi_level_seq_l1_col = "NPI Level Sequence L1"
    npi_item_level_col = "NPI Item Level"
    npi_account_level_col = "NPI Account Level"
    npi_channel_level_col = "NPI Channel Level"
    npi_region_level_col = "NPI Region Level"
    npi_pnl_level_col = "NPI PnL Level"
    npi_location_level_col = "NPI Location Level"
    npi_dd_level_col = "NPI Demand Domain Level"
    assortment_final_col = "Assortment Final"

    # output columns
    cols_required_in_output_npi_ass_l0 = [
        version_col,
        initiative_col,
        data_object_col,
        npi_item_col,
        npi_account_col,
        npi_channel_col,
        npi_region_col,
        npi_pnl_col,
        npi_location_col,
        npi_demand_domain_col,
        npi_association_l0_col,
    ]

    cols_required_in_output_npi_ass_final_bylev = [
        version_col,
        initiative_col,
        data_object_col,
        item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        loc_col,
        npi_assorment_final_by_level_col,
    ]

    cols_required_in_output_npi_ass_final = [
        version_col,
        initiative_col,
        item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        loc_col,
        npi_assortment_final_col,
    ]

    cols_required_in_output_pl_level_assortment = [
        version_col,
        initiative_col,
        data_object_col,
        pl_item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        planning_location_col,
        pl_level_assortmnet_col,
        sys_split_perc_col,
    ]

    # Output empty dataframes
    npi_association_l0_empty = pd.DataFrame(columns=cols_required_in_output_npi_ass_l0)
    npi_association_final_by_level_empty = pd.DataFrame(
        columns=cols_required_in_output_npi_ass_final_bylev
    )
    npi_association_final_empty = pd.DataFrame(columns=cols_required_in_output_npi_ass_final)
    pl_level_assortment_empty = pd.DataFrame(columns=cols_required_in_output_pl_level_assortment)

    # Combined empty output dataframes
    combined_output_dataframes = [
        npi_association_l0_empty,
        npi_association_final_by_level_empty,
        npi_association_final_empty,
        pl_level_assortment_empty,
    ]

    try:

        if df_keys is None:
            df_keys = {}

        # assert and convert string value to boolean
        assert ReadFromHive in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(ReadFromHive)
        ReadFromHive = eval(ReadFromHive)

        # Multiple versions do not support
        if len(InitiativeLevel[version_col].unique()) >= 2:
            logger.error(
                "The plugin does not support multiple versions. The user selected versions are: {}".format(
                    InitiativeLevel[version_col].unique()
                )
            )

        if pd.isna(AssortmentType) or AssortmentType in [np.nan, "nan", "", None]:
            raise ValueError(
                f"AssortmentType is invalid. User input: {AssortmentType} for slice: {df_keys}."
            )

        # Value
        version = InitiativeLevel[version_col].values[0]

        # NPI Item Scope pre-processing
        NPIItemScope = split_to_list(NPIItemScope, Delimiter)

        if (AssortmentType != "FileUpload") & (NPIItemScope is None or len(NPIItemScope) == 0):
            logger.error(
                f"Invalid NPI Item Scope. Please select correct NPI Item for slice: {df_keys}."
            )

        # InitiativeLevel pre-processing
        InitiativeLevel.drop([version_col], axis=1, inplace=True)
        InitiativeLevel[npi_level_seq_l1_col] = pd.to_numeric(
            InitiativeLevel[npi_level_seq_l1_col], errors="coerce", downcast="integer"
        )

        # Remove null records
        InitiativeLevel = InitiativeLevel[InitiativeLevel[npi_level_seq_l1_col].notna()]

        # Sort the table based on sequence
        InitiativeLevel.sort_values(by=[npi_level_seq_l1_col], ascending=False, inplace=True)

        # AssortmentFinal pre-processing
        AssortmentFinal.drop([version_col], axis=1, inplace=True)
        AssortmentFinal[assortment_final_col] = pd.to_numeric(
            AssortmentFinal[assortment_final_col], errors="coerce", downcast="integer"
        )
        AssortmentFinal = AssortmentFinal[AssortmentFinal[assortment_final_col] > 0]

        if Level in [None, "na", "NA", "None", "", " "]:
            logger.error(
                f"Invalid user input Level. User input is {Level}. Accepted value is 'NPI Level #'"
            )

        if InitiativeLevel is None or len(InitiativeLevel) == 0:
            logger.error("'InitiativeLevel' input can't be empty for slice: {}".format(df_keys))
            return combined_output_dataframes
        # ==---------------------------------------- Calculate NPI Association L0
        logger.info("Calculating NPI Association L0 ...")

        if AssortmentFinal is None or len(AssortmentFinal) == 0:
            logger.error("'AssortmentFinal' input can't be empty for slice: {}".format(df_keys))
            return combined_output_dataframes

        # User specified level based on data object and Initiative
        InitiativeLevel_required = InitiativeLevel[
            (InitiativeLevel[data_object_col] == Level)
            & (InitiativeLevel[initiative_col] == Initiative)
        ]

        if InitiativeLevel_required is None or len(InitiativeLevel_required) == 0:
            logger.error("'InitiativeLevel_required' is empty for slice: {}".format(df_keys))
            return combined_output_dataframes

        # Get user defined level
        (
            user_item_level_col,
            user_account_level_col,
            user_channel_level_col,
            user_region_level_col,
            user_pnl_level_col,
            user_dd_level_col,
            user_loc_level_col,
        ) = user_selected_level(
            InitiativeLevel_required,
            npi_item_level_col,
            npi_account_level_col,
            npi_channel_level_col,
            npi_region_level_col,
            npi_pnl_level_col,
            npi_dd_level_col,
            npi_location_level_col,
        )

        # Select Relavant columns
        ItemMaster_req = ItemMaster[[item_col, pl_item_col, user_item_level_col]].drop_duplicates()
        AccountMaster_req = AccountMaster[
            [planning_account_col, user_account_level_col]
        ].drop_duplicates()
        ChannelMaster_req = ChannelMaster[
            [planning_channel_col, user_channel_level_col]
        ].drop_duplicates()
        RegionMaster_req = RegionMaster[
            [planning_region_col, user_region_level_col]
        ].drop_duplicates()
        PnLMaster_req = PnLMaster[[planning_pnl_col, user_pnl_level_col]].drop_duplicates()
        DemandDomainMaster_req = DemandDomainMaster[
            [planning_demand_domain_col, user_dd_level_col]
        ].drop_duplicates()
        LocationMaster_req = LocationMaster[
            [loc_col, planning_location_col, user_loc_level_col]
        ].drop_duplicates()

        # Remove duplicate cols
        ItemMaster_req = ItemMaster_req.loc[:, ~ItemMaster_req.columns.duplicated()]
        AccountMaster_req = AccountMaster_req.loc[:, ~AccountMaster_req.columns.duplicated()]
        ChannelMaster_req = ChannelMaster_req.loc[:, ~ChannelMaster_req.columns.duplicated()]
        RegionMaster_req = RegionMaster_req.loc[:, ~RegionMaster_req.columns.duplicated()]
        PnLMaster_req = PnLMaster_req.loc[:, ~PnLMaster_req.columns.duplicated()]
        DemandDomainMaster_req = DemandDomainMaster_req.loc[
            :, ~DemandDomainMaster_req.columns.duplicated()
        ]
        LocationMaster_req = LocationMaster_req.loc[:, ~LocationMaster_req.columns.duplicated()]

        if AssortmentType == "InitiateAssortment":
            #   --- Case 1

            # Get the lower level assortment data
            InitiativeLevel.sort_values(by=npi_level_seq_l1_col, ascending=True, inplace=True)
            NPIAssortmentFinalByLevel_relevant = get_lower_level_ass_data(
                NPIAssortmentFinalByLevel,
                Initiative,
                Level,
                InitiativeLevel,
                initiative_col,
                data_object_col,
            )

            # Merge with the assortment to get all the required columns
            Master_list = [
                (item_col, ItemMaster_req),
                (planning_account_col, AccountMaster_req),
                (planning_channel_col, ChannelMaster_req),
                (planning_region_col, RegionMaster_req),
                (planning_pnl_col, PnLMaster_req),
                (planning_demand_domain_col, DemandDomainMaster_req),
                (loc_col, LocationMaster_req),
            ]

            AssortmentFinal_req = reduce(merge_two, Master_list, NPIAssortmentFinalByLevel_relevant)

            # Data validation
            NPIAccountScope = split_to_list(NPIAccountScope, Delimiter)
            NPIChannelScope = split_to_list(NPIChannelScope, Delimiter)
            NPIRegionScope = split_to_list(NPIRegionScope, Delimiter)
            NPIPnLScope = split_to_list(NPIPnLScope, Delimiter)
            NPIDemandDomainScope = split_to_list(NPIDemandDomainScope, Delimiter)
            NPILocationScope = split_to_list(NPILocationScope, Delimiter)

            # Filter based on given scope
            AssortmentFinal_req = AssortmentFinal_req[
                AssortmentFinal_req[user_item_level_col].isin(NPIItemScope)
                & AssortmentFinal_req[user_account_level_col].isin(NPIAccountScope)
                & AssortmentFinal_req[user_channel_level_col].isin(NPIChannelScope)
                & AssortmentFinal_req[user_region_level_col].isin(NPIRegionScope)
                & AssortmentFinal_req[user_pnl_level_col].isin(NPIPnLScope)
                & AssortmentFinal_req[user_dd_level_col].isin(NPIDemandDomainScope)
                & AssortmentFinal_req[user_loc_level_col].isin(NPILocationScope)
            ]

            if AssortmentFinal_req is None or len(AssortmentFinal_req) == 0:
                logger.error("'AssortmentFinal_req' is empty for slice: {}".format(df_keys))
                return combined_output_dataframes

            # Get the active assortments from the similar item and apply the same to the NPI item
            npi_active_intersections = AssortmentFinal_req[
                [
                    user_account_level_col,
                    user_channel_level_col,
                    user_region_level_col,
                    user_pnl_level_col,
                    user_dd_level_col,
                    user_loc_level_col,
                    planning_account_col,
                    planning_channel_col,
                    planning_region_col,
                    planning_pnl_col,
                    planning_demand_domain_col,
                    planning_location_col,
                    loc_col,
                ]
            ].drop_duplicates()
            npi_active_intersections = npi_active_intersections.loc[
                :, ~npi_active_intersections.columns.duplicated()
            ]

            # Make the active intersections for NPI | Cross product
            npi_active_intersections = npi_active_intersections.reset_index(drop=True)
            NPIItemScope_df = pd.DataFrame(NPIItemScope, columns=[user_item_level_col])
            NPIItemScope_df = pd.merge(
                NPIItemScope_df, ItemMaster_req, on=[user_item_level_col], how="left"
            )

            grainular_level_assortments = create_cartesian_product(
                NPIItemScope_df, npi_active_intersections
            )

        elif AssortmentType == "AddAssortment":
            #   --- Case 2: If SimilarItemScope do not available
            logger.info(
                "Similar Item Scope is missing. Assortment will be created with all the possibilities ..."
            )

            # Data validation
            NPIAccountScope = split_to_list(NPIAccountScope, Delimiter)
            NPIChannelScope = split_to_list(NPIChannelScope, Delimiter)
            NPIRegionScope = split_to_list(NPIRegionScope, Delimiter)
            NPIPnLScope = split_to_list(NPIPnLScope, Delimiter)
            NPIDemandDomainScope = split_to_list(NPIDemandDomainScope, Delimiter)
            NPILocationScope = split_to_list(NPILocationScope, Delimiter)

            if NPIAccountScope is None or len(NPIAccountScope) == 0:
                logger.error(
                    "Similar Item scope and NPI Account Scope are not available/valid for slice {}...".format(
                        df_keys
                    )
                )
                logger.error("Returning empty dataframes ...")
                return combined_output_dataframes

            if NPIChannelScope is None or len(NPIChannelScope) == 0:
                logger.error(
                    "Similar Item scope and NPI Channel Scope are not available/valid for slice {}...".format(
                        df_keys
                    )
                )
                logger.error("Returning empty dataframes ...")
                return combined_output_dataframes

            if NPIRegionScope is None or len(NPIRegionScope) == 0:
                logger.error(
                    "Similar Item scope and NPI Region Scope are not available/valid for slice {}...".format(
                        df_keys
                    )
                )
                logger.error("Returning empty dataframes ...")
                return combined_output_dataframes

            if NPIPnLScope is None or len(NPIPnLScope) == 0:
                logger.error(
                    "Similar Item scope and NPI PnL Scope are not available/valid for slice {}...".format(
                        df_keys
                    )
                )
                logger.error("Returning empty dataframes ...")
                return combined_output_dataframes

            if NPIDemandDomainScope is None or len(NPIDemandDomainScope) == 0:
                logger.error(
                    "Similar Item scope and NPI Demand Domain Scope are not available/valid for slice {}...".format(
                        df_keys
                    )
                )
                logger.error("Returning empty dataframes ...")
                return combined_output_dataframes

            if NPILocationScope is None or len(NPILocationScope) == 0:
                logger.error(
                    "Similar Item scope and NPI Location Scope are not available/valid for slice {}...".format(
                        df_keys
                    )
                )
                logger.error("Returning empty dataframes ...")
                return combined_output_dataframes

            # Convert to dataframe
            NPIItemScope_df = pd.DataFrame(NPIItemScope, columns=[user_item_level_col])
            NPIAccountScope_df = pd.DataFrame(NPIAccountScope, columns=[user_account_level_col])
            NPIChannelScope_df = pd.DataFrame(NPIChannelScope, columns=[user_channel_level_col])
            NPIRegionScope_df = pd.DataFrame(NPIRegionScope, columns=[user_region_level_col])
            NPIPnLScope_df = pd.DataFrame(NPIPnLScope, columns=[user_pnl_level_col])
            NPIDemandDomainScope_df = pd.DataFrame(
                NPIDemandDomainScope, columns=[user_dd_level_col]
            )
            NPILocationScope_df = pd.DataFrame(NPILocationScope, columns=[user_loc_level_col])

            # Get the lower level(s)
            NPIItemScope_df = pd.merge(
                NPIItemScope_df, ItemMaster_req, on=[user_item_level_col], how="left"
            )
            NPIAccountScope_df = pd.merge(
                NPIAccountScope_df, AccountMaster_req, on=[user_account_level_col], how="left"
            )
            NPIChannelScope_df = pd.merge(
                NPIChannelScope_df, ChannelMaster_req, on=[user_channel_level_col], how="left"
            )
            NPIRegionScope_df = pd.merge(
                NPIRegionScope_df, RegionMaster_req, on=[user_region_level_col], how="left"
            )
            NPIPnLScope_df = pd.merge(
                NPIPnLScope_df, PnLMaster_req, on=[user_pnl_level_col], how="left"
            )
            NPIDemandDomainScope_df = pd.merge(
                NPIDemandDomainScope_df, DemandDomainMaster_req, on=[user_dd_level_col], how="left"
            )
            NPILocationScope_df = pd.merge(
                NPILocationScope_df, LocationMaster_req, on=[user_loc_level_col], how="left"
            )

            # Cross product | make all possible intersections
            grainular_level_assortments = create_cartesian_product(
                NPIItemScope_df, NPIAccountScope_df
            )
            grainular_level_assortments = create_cartesian_product(
                grainular_level_assortments, NPIChannelScope_df
            )
            grainular_level_assortments = create_cartesian_product(
                grainular_level_assortments, NPIRegionScope_df
            )
            grainular_level_assortments = create_cartesian_product(
                grainular_level_assortments, NPIPnLScope_df
            )
            grainular_level_assortments = create_cartesian_product(
                grainular_level_assortments, NPIDemandDomainScope_df
            )
            grainular_level_assortments = create_cartesian_product(
                grainular_level_assortments, NPILocationScope_df
            )

        elif AssortmentType == "FileUpload":
            # Since NPIAssortmentFileUpload is already at npi level, just rename for npi_association_l0_output
            # We need to create grainular_level_assortments to prepare npi assortment final by level output

            if len(NPIAssortmentFileUpload) == 0:
                logger.error(
                    f"NPIAssortmentFileUpload can't be empty if AssortmentType is {AssortmentType} ..."
                )

            # Dropping email col
            NPIAssortmentFileUpload.drop(columns=[email_col], axis=1, inplace=True)

            # Creating npi association l0 col
            NPIAssortmentFileUpload[npi_association_l0_col] = 1

            # Output data frame npi_association_l0_output
            npi_association_l0_output = (
                NPIAssortmentFileUpload[cols_required_in_output_npi_ass_l0]
                .drop_duplicates()
                .copy(deep=True)
            )

            # Creating grainular level assortments from NPIAssortmentFileUpload

            # Map the NPI column to the original columns
            NPIAssortmentFileUpload.rename(
                columns={
                    npi_item_col: user_item_level_col,
                    npi_account_col: user_account_level_col,
                    npi_channel_col: user_channel_level_col,
                    npi_region_col: user_region_level_col,
                    npi_pnl_col: user_pnl_level_col,
                    npi_demand_domain_col: user_dd_level_col,
                    npi_location_col: user_loc_level_col,
                },
                inplace=True,
            )

            # Merge with the assortment to get all the required columns
            Master_list = [
                (user_item_level_col, ItemMaster_req),
                (user_account_level_col, AccountMaster_req),
                (user_channel_level_col, ChannelMaster_req),
                (user_region_level_col, RegionMaster_req),
                (user_pnl_level_col, PnLMaster_req),
                (user_dd_level_col, DemandDomainMaster_req),
                (user_loc_level_col, LocationMaster_req),
            ]

            grainular_level_assortments = reduce(merge_two, Master_list, NPIAssortmentFileUpload)

        # Preparing output dataframe if not fileupload since fileupload already created the output file
        if AssortmentType != "FileUpload":
            # Assortment at user specified level
            npi_association_l0_output = grainular_level_assortments[
                [
                    user_item_level_col,
                    user_account_level_col,
                    user_channel_level_col,
                    user_region_level_col,
                    user_pnl_level_col,
                    user_dd_level_col,
                    user_loc_level_col,
                ]
            ].drop_duplicates()

            # Required informations
            npi_association_l0_output[initiative_col] = Initiative
            npi_association_l0_output[npi_association_l0_col] = 1
            npi_association_l0_output[version_col] = version
            npi_association_l0_output[data_object_col] = Level

            # Map columns
            npi_association_l0_output = npi_association_l0_output.rename(
                columns={
                    user_item_level_col: npi_item_col,
                    user_account_level_col: npi_account_col,
                    user_channel_level_col: npi_channel_col,
                    user_region_level_col: npi_region_col,
                    user_pnl_level_col: npi_pnl_col,
                    user_dd_level_col: npi_demand_domain_col,
                    user_loc_level_col: npi_location_col,
                }
            )

            # Required columns for the output
            npi_association_l0_output = npi_association_l0_output[
                cols_required_in_output_npi_ass_l0
            ]

        if npi_association_l0_output is None or len(npi_association_l0_output) == 0:
            logger.error("'npi_association_l0_output' is empty for slice: {}".format(df_keys))
            return combined_output_dataframes

        logger.info("Calculated NPI Association L0 ")

        #  .......................................... End: Calculate NPI Association L0

        # ===---------------------------------------- Calculate NPI Assortment Final by Level
        logger.info("Calculating NPI Assortment Final by Level ...")

        if AssortmentType == "InitiateAssortment":

            #   --- Case 1: If SimilarItemScope available

            # Assortment at lower level
            npi_assortment_final_by_level_output = grainular_level_assortments[
                [
                    item_col,
                    planning_account_col,
                    planning_channel_col,
                    planning_region_col,
                    planning_pnl_col,
                    planning_demand_domain_col,
                    loc_col,
                ]
            ].drop_duplicates()

        else:
            #   --- Case 2: If SimilarItemScope not available
            # Create assortment from Assortment Final if intersections from output 'NPI Association L0' is available in AssortmentFinal
            # If the intersection not present then drop the columns by AssortmentSequence and so on until get the data. Then cross product the data with dropped columns.

            AssortmentSequence = split_to_list(AssortmentSequence, Delimiter)

            logger.info("Executing polar method for NPI Assortment Final by Level...")
            npi_assortment_final_by_level_output = find_intersections_using_polar(
                # User col names info
                user_item_level_col,
                user_account_level_col,
                user_channel_level_col,
                user_region_level_col,
                user_pnl_level_col,
                user_dd_level_col,
                user_loc_level_col,
                AssortmentSequence,
                # DataFrames
                AssortmentFinal,
                grainular_level_assortments,
                df_keys,
            )

        # Required info
        npi_assortment_final_by_level_output[version_col] = version
        npi_assortment_final_by_level_output[initiative_col] = Initiative
        npi_assortment_final_by_level_output[data_object_col] = Level
        npi_assortment_final_by_level_output[npi_assorment_final_by_level_col] = 1

        # Required columns for output
        npi_assortment_final_by_level_output = npi_assortment_final_by_level_output[
            cols_required_in_output_npi_ass_final_bylev
        ]

        if (
            npi_assortment_final_by_level_output is None
            or len(npi_assortment_final_by_level_output) == 0
        ):
            logger.error(
                "'npi_assortment_final_by_level_output' is empty for slice: {}".format(df_keys)
            )
            return combined_output_dataframes

        logger.info("Calculated NPI Assortment Final by Level.")

        #  .......................................... End: Calculate NPI Assortment Final by Level

        # ===---------------------------------------- Calculate NPI Assortment Final
        logger.info("Calculating NPI Assortment Final ...")

        NPIAssortmentFinalByLevel_filtered = NPIAssortmentFinalByLevel[
            (NPIAssortmentFinalByLevel[initiative_col] == Initiative)
        ]

        # Concat NPIAssortmentFinalByLevel with npi_assortment_final_by_level_output
        NPIAssortmentFinalByLevel_combined = pd.concat(
            [npi_assortment_final_by_level_output, NPIAssortmentFinalByLevel_filtered],
            ignore_index=True,
        )

        # Get level having assortment data based on high sequence in 'InitiativeLevel'
        InitiativeLevel.sort_values(by=npi_level_seq_l1_col, ascending=False, inplace=True)
        level_with_data = get_level_contains_data(
            NPIAssortmentFinalByLevel_combined, InitiativeLevel, data_object_col
        )

        if level_with_data is None:
            logger.error(
                "No Level found which contains assortment final by level data for slice {} ...".format(
                    df_keys
                )
            )
            logger.error("Returning empty dataframe...")
            return combined_output_dataframes

        # Current level will be active
        NPIAssortmentFinalByLevel_combined = NPIAssortmentFinalByLevel_combined[
            NPIAssortmentFinalByLevel_combined[data_object_col] == level_with_data
        ]

        NPIAssortmentFinalByLevel_combined.drop(
            [data_object_col, npi_assorment_final_by_level_col], axis=1, inplace=True
        )
        NPIAssortmentFinalByLevel_combined.drop_duplicates(inplace=True)

        # Activate the intersections
        NPIAssortmentFinalByLevel_combined[npi_assortment_final_col] = 1

        # Deactivate existing assortment for the Initiative
        NPIAssortmentFinal_inactive = NPIAssortmentFinal[
            NPIAssortmentFinal[initiative_col] == Initiative
        ].copy(deep=True)
        NPIAssortmentFinal_inactive[npi_assortment_final_col] = np.nan

        npi_assortment_final_output = pd.concat(
            [NPIAssortmentFinalByLevel_combined, NPIAssortmentFinal_inactive], ignore_index=True
        )

        # Required columns for the output
        npi_assortment_final_output = npi_assortment_final_output.groupby(
            [
                version_col,
                initiative_col,
                item_col,
                planning_account_col,
                planning_channel_col,
                planning_region_col,
                planning_pnl_col,
                planning_demand_domain_col,
                loc_col,
            ],
            as_index=False,
        )[npi_assortment_final_col].max()
        npi_assortment_final_output = npi_assortment_final_output[
            cols_required_in_output_npi_ass_final
        ]

        if npi_assortment_final_output is None or len(npi_assortment_final_output) == 0:
            logger.error("'npi_assortment_final_output' is empty for slice: {}".format(df_keys))
            return combined_output_dataframes

        logger.info("Calculated NPI Assortment Final.")

        #  .......................................... End: Calculate NPI Assortment Final

        # ===---------------------------------------- Calculate NPI Planning Assortment by Level
        logger.info("Calculating NPI Planning Assortment by Level ...")

        # --- Aggregate output 2 -> group using output 1 -> calculate split %
        # Filter grainular_level_assortments for output 2
        common_col = list(
            set(grainular_level_assortments.columns)
            & set(npi_assortment_final_by_level_output.columns)
        )
        grainular_level_assortments_active = pd.merge(
            grainular_level_assortments,
            npi_assortment_final_by_level_output[common_col].drop_duplicates(),
            on=common_col,
            how="inner",
        )

        # Group based on Initiative level provided by user, and disaggregate sys split %
        Group = grainular_level_assortments_active.groupby(
            [
                user_item_level_col,
                user_account_level_col,
                user_channel_level_col,
                user_region_level_col,
                user_pnl_level_col,
                user_loc_level_col,
                user_dd_level_col,
            ]
        )
        result_list = []
        for _, the_data in Group:

            # Selecting the req cols
            the_data = the_data[
                [
                    pl_item_col,
                    planning_account_col,
                    planning_channel_col,
                    planning_region_col,
                    planning_pnl_col,
                    planning_demand_domain_col,
                    planning_location_col,
                ]
            ].drop_duplicates()

            sys_split_perc: float
            if len(the_data) == 0:
                sys_split_perc = 0.0
            else:
                sys_split_perc = 1 / len(the_data)

            the_data[sys_split_perc_col] = sys_split_perc

            result_list.append(the_data)

        # Unpack
        pl_level_assortment_output = concat_to_dataframe(result_list)
        pl_level_assortment_output[pl_level_assortmnet_col] = 1

        # Required info
        pl_level_assortment_output[version_col] = version
        pl_level_assortment_output[initiative_col] = Initiative
        pl_level_assortment_output[data_object_col] = Level

        # Required columns for the output
        pl_level_assortment_output = pl_level_assortment_output[
            cols_required_in_output_pl_level_assortment
        ]

        if pl_level_assortment_output is None or len(pl_level_assortment_output) == 0:
            logger.error("'pl_level_assortment_output' is empty for slice: {}".format(df_keys))
            return combined_output_dataframes

        logger.info("Calculated NPI Association L0 ")

        #  .......................................... End: Calculate NPI Planning Assortment by Level

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        logger.exception(e)
        return combined_output_dataframes

    logger.info("Successfully Executed DP123CreateNPIAssortment")

    return (
        npi_association_l0_output,
        npi_assortment_final_by_level_output,
        npi_assortment_final_output,
        pl_level_assortment_output,
    )


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    # Params
    Initiative: Optional[str] = None,
    Level: Optional[str] = None,
    AssortmentType: Optional[str] = None,
    AssortmentSequence: Optional[str] = None,
    # NPI Scope
    NPIItemScope: Optional[str] = None,
    NPIAccountScope: Optional[str] = None,
    NPIChannelScope: Optional[str] = None,
    NPIRegionScope: Optional[str] = None,
    NPIPnLScope: Optional[str] = None,
    NPIDemandDomainScope: Optional[str] = None,
    NPILocationScope: Optional[str] = None,
    # Data
    AssortmentFinal: pd.DataFrame = None,
    NPIAssortmentFileUpload: pd.DataFrame = None,
    InitiativeLevel: pd.DataFrame = None,
    NPIAssortmentFinal: pd.DataFrame = None,
    NPIAssortmentFinalByLevel: pd.DataFrame = None,
    # Master data
    ItemMaster: pd.DataFrame = None,
    AccountMaster: pd.DataFrame = None,
    ChannelMaster: pd.DataFrame = None,
    RegionMaster: pd.DataFrame = None,
    PnLMaster: pd.DataFrame = None,
    DemandDomainMaster: pd.DataFrame = None,
    LocationMaster: pd.DataFrame = None,
    # Others
    Delimiter: str = ",",
    ReadFromHive: str = "False",
    df_keys: Optional[dict] = None,
):
    """Entry point of the script for a single initiative and level."""
    plugin_name = "DP123CreateNPIAssortment"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

    # Configuration
    initiative_col = "Initiative.[Initiative]"
    data_object_col = "Data Object.[Data Object]"

    if AssortmentType == "FileUpload":
        # In case of file upload iterate to each unique initiative and level, calculate the output using the single_initiative_and_level function and append all the output at the end

        logger.info(
            "AssortmentType is FileUpload, iterating through each unique initiative and level ..."
        )

        NPIAssortmentFileUpload_Group = NPIAssortmentFileUpload.groupby(
            [initiative_col, data_object_col]
        )

        NPIAssociationL0Output_list = []
        NPIAssortmentFinalByLevelOutput_list = []
        NPIAssortmentFinalOutput_list = []
        PlLevelAssortmentOutput_list = []

        for key, group in NPIAssortmentFileUpload_Group:

            # Current Initiative and level
            Initiative = key[0]
            Level = key[1]

            (
                NPIAssociationL0Output_single,
                NPIAssortmentFinalByLevelOutput_single,
                NPIAssortmentFinalOutput_single,
                PlLevelAssortmentOutput_single,
            ) = single_initiative_and_level(
                # Params
                Initiative=Initiative,
                Level=Level,
                AssortmentType=AssortmentType,
                AssortmentSequence=AssortmentSequence,
                # NPI Scope
                NPIItemScope=NPIItemScope,
                NPIAccountScope=NPIAccountScope,
                NPIChannelScope=NPIChannelScope,
                NPIRegionScope=NPIRegionScope,
                NPIPnLScope=NPIPnLScope,
                NPIDemandDomainScope=NPIDemandDomainScope,
                NPILocationScope=NPILocationScope,
                # Data
                AssortmentFinal=AssortmentFinal.copy(deep=True),
                NPIAssortmentFileUpload=group.copy(
                    deep=True
                ),  # NPI AssortmentFileUpload is grouped by Initiative and Level
                InitiativeLevel=InitiativeLevel.copy(deep=True),
                NPIAssortmentFinal=NPIAssortmentFinal.copy(deep=True),
                NPIAssortmentFinalByLevel=NPIAssortmentFinalByLevel.copy(deep=True),
                # Master data
                ItemMaster=ItemMaster.copy(deep=True),
                AccountMaster=AccountMaster.copy(deep=True),
                ChannelMaster=ChannelMaster.copy(deep=True),
                RegionMaster=RegionMaster.copy(deep=True),
                PnLMaster=PnLMaster.copy(deep=True),
                DemandDomainMaster=DemandDomainMaster.copy(deep=True),
                LocationMaster=LocationMaster.copy(deep=True),
                # Others
                Delimiter=Delimiter,
                ReadFromHive=ReadFromHive,
                df_keys=df_keys,
            )

            # Append each output
            NPIAssociationL0Output_list.append(NPIAssociationL0Output_single)
            NPIAssortmentFinalByLevelOutput_list.append(NPIAssortmentFinalByLevelOutput_single)
            NPIAssortmentFinalOutput_list.append(NPIAssortmentFinalOutput_single)
            PlLevelAssortmentOutput_list.append(PlLevelAssortmentOutput_single)

        NPIAssociationL0Output = concat_to_dataframe(NPIAssociationL0Output_list)
        NPIAssortmentFinalByLevelOutput = concat_to_dataframe(NPIAssortmentFinalByLevelOutput_list)
        NPIAssortmentFinalOutput = concat_to_dataframe(NPIAssortmentFinalOutput_list)
        PlLevelAssortmentOutput = concat_to_dataframe(PlLevelAssortmentOutput_list)

    else:
        # In case the plugin runs for a single initiative and level, we can directly call the single_initiative_and_level function

        logger.info(
            f"AssortmentType is {AssortmentType}, executing for single initiative and level ..."
        )

        (
            NPIAssociationL0Output,
            NPIAssortmentFinalByLevelOutput,
            NPIAssortmentFinalOutput,
            PlLevelAssortmentOutput,
        ) = single_initiative_and_level(
            # Params
            Initiative=Initiative,
            Level=Level,
            AssortmentType=AssortmentType,
            AssortmentSequence=AssortmentSequence,
            # NPI Scope
            NPIItemScope=NPIItemScope,
            NPIAccountScope=NPIAccountScope,
            NPIChannelScope=NPIChannelScope,
            NPIRegionScope=NPIRegionScope,
            NPIPnLScope=NPIPnLScope,
            NPIDemandDomainScope=NPIDemandDomainScope,
            NPILocationScope=NPILocationScope,
            # Data
            AssortmentFinal=AssortmentFinal,
            NPIAssortmentFileUpload=NPIAssortmentFileUpload,
            InitiativeLevel=InitiativeLevel,
            NPIAssortmentFinal=NPIAssortmentFinal,
            NPIAssortmentFinalByLevel=NPIAssortmentFinalByLevel,
            # Master data
            ItemMaster=ItemMaster,
            AccountMaster=AccountMaster,
            ChannelMaster=ChannelMaster,
            RegionMaster=RegionMaster,
            PnLMaster=PnLMaster,
            DemandDomainMaster=DemandDomainMaster,
            LocationMaster=LocationMaster,
            # Others
            Delimiter=Delimiter,
            ReadFromHive=ReadFromHive,
            df_keys=df_keys,
        )

    return (
        NPIAssociationL0Output,
        NPIAssortmentFinalByLevelOutput,
        NPIAssortmentFinalOutput,
        PlLevelAssortmentOutput,
    )
