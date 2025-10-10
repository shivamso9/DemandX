"""Publish NPI Forecast to Consensus for Flexible NPI.

Pseudocode:
    Version: 2025.04.00
    --------------------
        -Step1: Iterate for each rows of 'SelectedCombinations' input
        -Step2: Get the user selected level from 'Initiative levels' input
        -Step3: Identify the assortment for which the forecast needs to be published from 'Parameters' input
            - Filter: Current Date < Independence Date
        -Step4: Null out the assortment for selected Initiative using input table : 'NPIAssortmentFinal' input
        -Step5: Populate the Assortment measure where split % is not null for selected Initiative and Level
        -Step6: Populate the dates at planning level from NPI level using NPI Level Planning Level Mapping process
            - Filter:
                - Current Initiative and Level
                - Current Date < Independence Date
                - split % > 0
        -Step7: Remove the Initiative and Level and aggregate based on the intersection of
            -> Output
        -Step8: If Assortment New is 1 then null out the Assortment Inactive
            -> Output
        -Step9: Give the output of Intro Date and Disco date by droping the Item and Location column for selected Initiative and Level(nulled value should not be in output) using [Step6]
            -> Output
        -Step10: For the assorted intersections from [Step6] populate NPI publish Fcst.
            -> Output
"""

import logging
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
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


def to_datetime_safely(df, col, format=None):
    """Convert the date time column safely."""
    if format in [None, "None", "", "nan", "np.nan"]:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception as e:
            logger.error(f"Unsupported date format for column: {col}. Please specify the 'format'.")
            raise ValueError(f"Invalid {col} format for input columns: {df.columns}. Error: {e}")
    else:
        try:
            df[col] = pd.to_datetime(df[col], format=format)
        except ValueError as e:
            try:
                logger.warning(f"{col} conversion has failed using format: {format}. Error: {e}...")
                logger.warning(f"Reattempting the {col} conversion with default date format...")
                df[col] = pd.to_datetime(df[col])
                logger.info(f"Successfully converted the {col} format.")
            except ValueError as ex:
                logger.error(f"Cannot convert the {col}. Error: {ex}")
                raise ValueError("Invalid {col} format for input columns: {df.columns}.")
    return df


col_mapping = {
    "NPI Fcst Published": float,
    "Intro Date": "datetime64[ns]",
    "Disco Date": "datetime64[ns]",
    "Assortment New": float,
    "Assortment Inactive": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    # Data
    SelectedCombinations: pd.DataFrame = None,
    InitiativeLevels: pd.DataFrame = None,
    Parameters: pd.DataFrame = None,
    NPIFcst: pd.DataFrame = None,
    # Master data
    ItemMaster: pd.DataFrame = None,
    AccountMaster: pd.DataFrame = None,
    ChannelMaster: pd.DataFrame = None,
    RegionMaster: pd.DataFrame = None,
    PnLMaster: pd.DataFrame = None,
    DemandDomainMaster: pd.DataFrame = None,
    LocationMaster: pd.DataFrame = None,
    # Others
    Splits: pd.DataFrame = None,
    NPIAssortmentFinal: pd.DataFrame = None,
    CurrentDate: pd.DataFrame = None,
    MeasureDateFormat=None,
    TimeKeyFormat=None,
    df_keys: Optional[dict] = None,
):
    """Entry point of the script."""
    plugin_name = "DP131PublishNPIForecast"
    logger.info("Executing {} for slice {}".format(plugin_name, df_keys))

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
    daykey_col = "Time.[DayKey]"

    npi_association_l0_col = "NPI Association L0"
    npi_assorment_final_by_level_col = "NPI Assortment Final by Level"
    sys_split_perc_col = "NPI Final Split %"
    npi_item_level_col = "NPI Item Level"
    npi_account_level_col = "NPI Account Level"
    npi_channel_level_col = "NPI Channel Level"
    npi_region_level_col = "NPI Region Level"
    npi_pnl_level_col = "NPI PnL Level"
    npi_location_level_col = "NPI Location Level"
    npi_dd_level_col = "NPI Demand Domain Level"
    npi_fcst_publish_flag_col = "NPI Forecast Publish Flag L1"
    start_date_col = "Start Date L0"
    independence_date_col = "Independence Date Final L0"
    assortment_new_col = "Assortment New"
    assortment_inactive_col = "Assortment Inactive"
    end_date_col = "End Date L0"
    intro_date_col = "Intro Date"
    disco_date_col = "Disco Date"
    npi_fcst_published_col = "NPI Fcst Published"
    pl_level_npi_fcst_col = "Planning Level NPI Fcst"
    partial_week_col = "Time.[Partial Week]"
    assortment_npi_col = "Assortment NPI"

    # Assume
    start_date_format = MeasureDateFormat
    independence_date_format = MeasureDateFormat

    # output columns
    cols_required_in_output_ass_new = [
        version_col,
        item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        loc_col,
        planning_demand_domain_col,
        assortment_new_col,
        assortment_npi_col,
    ]

    cols_required_in_output_ass_inactive = [
        version_col,
        item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        loc_col,
        planning_demand_domain_col,
        assortment_inactive_col,
    ]

    cols_required_in_output_dates = [
        version_col,
        pl_item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        planning_location_col,
        intro_date_col,
        disco_date_col,
    ]

    cols_required_in_output_npi_fcst = [
        version_col,
        pl_item_col,
        planning_account_col,
        planning_channel_col,
        planning_region_col,
        planning_pnl_col,
        planning_demand_domain_col,
        planning_location_col,
        partial_week_col,
        npi_fcst_published_col,
    ]

    # Output empty dataframes
    npi_fcst_output = pd.DataFrame(columns=cols_required_in_output_npi_fcst)
    dates_output = pd.DataFrame(columns=cols_required_in_output_dates)
    assortment_new_output = pd.DataFrame(columns=cols_required_in_output_ass_new)
    assortment_inactive_output = pd.DataFrame(columns=cols_required_in_output_ass_inactive)

    try:
        logger.info("Validating the data for slice: {}".format(df_keys))

        if CurrentDate is None or len(CurrentDate) == 0:
            raise ValueError("Please pass a valid current date.")

        CurrentDate = to_datetime_safely(CurrentDate, daykey_col, format=TimeKeyFormat)
        CurrentDate = CurrentDate[daykey_col].max()

        SelectedCombinations[npi_fcst_publish_flag_col] = pd.to_numeric(
            SelectedCombinations[npi_fcst_publish_flag_col], downcast="integer", errors="coerce"
        )
        SelectedCombinations = SelectedCombinations[
            SelectedCombinations[npi_fcst_publish_flag_col] >= 1
        ]

        if len(SelectedCombinations) == 0:
            raise ValueError(
                "'SelectedCombinations' input can't be empty for slice {}".format(df_keys)
            )

        if len(InitiativeLevels) == 0:
            raise ValueError(
                "'InitiativeLevels' input can't be empty for slice: {}".format(df_keys)
            )

        Parameters[npi_association_l0_col] = pd.to_numeric(
            Parameters[npi_association_l0_col], downcast="integer", errors="coerce"
        )
        Parameters = Parameters[Parameters[npi_association_l0_col] >= 1]
        Parameters = Parameters[Parameters[start_date_col].notna()]
        Parameters = Parameters[Parameters[independence_date_col].notna()]

        Parameters = to_datetime_safely(Parameters, start_date_col, format=start_date_format)
        Parameters = to_datetime_safely(
            Parameters, independence_date_col, format=independence_date_format
        )
        Parameters = Parameters[CurrentDate < Parameters[independence_date_col]]

        if len(Parameters) == 0:
            raise ValueError("'Parameters' input can't be empty for slice: {}".format(df_keys))

        if len(NPIFcst) == 0:
            raise ValueError("'NPIFcst' input can't be empty for slice: {}".format(df_keys))

        if len(Splits) == 0:
            raise ValueError("'Splits' input can't be empty for slice: {}".format(df_keys))

        NPIAssortmentFinal[npi_assorment_final_by_level_col] = pd.to_numeric(
            NPIAssortmentFinal[npi_assorment_final_by_level_col],
            downcast="integer",
            errors="coerce",
        )
        NPIAssortmentFinal = NPIAssortmentFinal[
            NPIAssortmentFinal[npi_assorment_final_by_level_col] >= 1
        ]

        if len(NPIAssortmentFinal) == 0:
            raise ValueError(
                "'NPIAssortmentFinal' input can't be empty for slice: {}".format(df_keys)
            )

        # Values
        version = SelectedCombinations[version_col].values[0]

        assortment_new_list = []
        dates_list = []
        npi_fcst_list = []

        # Iterate for SelectedCombinations
        for _, the_SelectedCombinations in SelectedCombinations.iterrows():

            the_InitiativeLevels = InitiativeLevels[
                (InitiativeLevels[initiative_col] == the_SelectedCombinations[initiative_col])
                & (InitiativeLevels[data_object_col] == the_SelectedCombinations[data_object_col])
            ]
            (
                the_item_level,
                the_account_level,
                the_channel_level,
                the_region_level,
                the_pnl_level,
                the_dd_level,
                the_loc_level,
            ) = user_selected_level(
                the_InitiativeLevels,
                npi_item_level_col,
                npi_account_level_col,
                npi_channel_level_col,
                npi_region_level_col,
                npi_pnl_level_col,
                npi_dd_level_col,
                npi_location_level_col,
            )

            # Relevant assortment
            the_Parameters = Parameters[
                (Parameters[initiative_col] == the_SelectedCombinations[initiative_col])
                & (Parameters[data_object_col] == the_SelectedCombinations[data_object_col])
            ]

            # For the current initiative
            master_list = [
                (pl_item_col, ItemMaster[[pl_item_col, the_item_level]].drop_duplicates()),
                (
                    planning_account_col,
                    AccountMaster[[planning_account_col, the_account_level]].drop_duplicates(),
                ),
                (
                    planning_channel_col,
                    ChannelMaster[[planning_channel_col, the_channel_level]].drop_duplicates(),
                ),
                (
                    planning_region_col,
                    RegionMaster[[planning_region_col, the_region_level]].drop_duplicates(),
                ),
                (planning_pnl_col, PnLMaster[[planning_pnl_col, the_pnl_level]].drop_duplicates()),
                (
                    planning_demand_domain_col,
                    DemandDomainMaster[
                        [planning_demand_domain_col, the_dd_level]
                    ].drop_duplicates(),
                ),
                (
                    planning_location_col,
                    LocationMaster[[planning_location_col, the_loc_level]].drop_duplicates(),
                ),
            ]
            NPIAssortmentFinal_w_master = reduce(merge_two, master_list, NPIAssortmentFinal)
            the_NPIAssortmentFinal = NPIAssortmentFinal_w_master[
                NPIAssortmentFinal_w_master[initiative_col]
                == the_SelectedCombinations[initiative_col]
            ]

            # Get params
            the_NPIAssortmentFinal = pd.merge(
                the_NPIAssortmentFinal,
                the_Parameters,
                left_on=[
                    initiative_col,
                    data_object_col,
                    the_item_level,
                    the_account_level,
                    the_channel_level,
                    the_region_level,
                    the_pnl_level,
                    the_loc_level,
                ],
                right_on=[
                    initiative_col,
                    data_object_col,
                    npi_item_col,
                    npi_account_col,
                    npi_channel_col,
                    npi_region_col,
                    npi_pnl_col,
                    npi_location_col,
                ],
                how="left",
            )

            if len(the_NPIAssortmentFinal) == 0:
                logger.warning(
                    "No records found in 'Parameters' for 'NPIAssortmentFinal' for slice: {} for intersection: {}".format(
                        df_keys, the_SelectedCombinations
                    )
                )
                logger.warning("Skipping to next iteration ...")
                continue

            key = [
                initiative_col,
                data_object_col,
                pl_item_col,
                planning_account_col,
                planning_channel_col,
                planning_region_col,
                planning_pnl_col,
                planning_demand_domain_col,
                planning_location_col,
            ]
            the_NPIAssortmentFinal = pd.merge(the_NPIAssortmentFinal, Splits, on=key, how="left")

            # assort for the selected initiative and level
            the_NPIAssortmentFinal[assortment_new_col] = np.where(
                (the_NPIAssortmentFinal[sys_split_perc_col] > 0)
                & (CurrentDate < the_NPIAssortmentFinal[independence_date_col])
                & (
                    the_NPIAssortmentFinal[data_object_col]
                    == the_SelectedCombinations[data_object_col]
                ),
                1,
                np.nan,
            )

            assortment_new = (
                the_NPIAssortmentFinal.groupby(
                    [
                        item_col,
                        planning_account_col,
                        planning_channel_col,
                        planning_region_col,
                        planning_pnl_col,
                        planning_demand_domain_col,
                        loc_col,
                    ]
                )[assortment_new_col]
                .max()
                .reset_index()
            )
            assortment_new_list.append(assortment_new)

            # --- Get intro date and disco date
            dates = the_NPIAssortmentFinal[the_NPIAssortmentFinal[assortment_new_col] == 1]
            dates.rename(
                columns={start_date_col: intro_date_col, end_date_col: disco_date_col}, inplace=True
            )
            dates = dates[
                [
                    pl_item_col,
                    planning_account_col,
                    planning_channel_col,
                    planning_region_col,
                    planning_pnl_col,
                    planning_demand_domain_col,
                    planning_location_col,
                    intro_date_col,
                    disco_date_col,
                ]
            ].drop_duplicates()

            dates_list.append(dates)

            # --- NPI Fcst published
            # For the initiative all the intersection will be the part of output either with data or with null
            the_NPIFcst = NPIFcst[
                NPIFcst[initiative_col] == the_SelectedCombinations[initiative_col]
            ]

            valid_the_NPIAssortmentFinal = the_NPIAssortmentFinal[
                the_NPIAssortmentFinal[assortment_new_col] == 1
            ]

            key = [
                initiative_col,
                data_object_col,
                pl_item_col,
                planning_account_col,
                planning_channel_col,
                planning_region_col,
                planning_pnl_col,
                planning_demand_domain_col,
                planning_location_col,
            ]
            valid_the_NPIAssortmentFinal = valid_the_NPIAssortmentFinal[
                key + [assortment_new_col]
            ].drop_duplicates()
            the_NPIFcst = pd.merge(the_NPIFcst, valid_the_NPIAssortmentFinal, on=key, how="left")

            # If assortment is active then populate fcst
            the_NPIFcst[npi_fcst_published_col] = the_NPIFcst[pl_level_npi_fcst_col].where(
                the_NPIFcst[assortment_new_col] == 1, np.nan
            )

            the_NPIFcst = (
                the_NPIFcst.groupby(
                    [
                        pl_item_col,
                        planning_account_col,
                        planning_channel_col,
                        planning_region_col,
                        planning_pnl_col,
                        planning_demand_domain_col,
                        planning_location_col,
                        partial_week_col,
                    ]
                )[npi_fcst_published_col]
                .agg(lambda x: x.sum() if x.notna().any() else pd.NA)
                .reset_index()
            )

            npi_fcst_list.append(the_NPIFcst)

        # --- Output formattion
        assortment_new_output = concat_to_dataframe(assortment_new_list)
        assortment_new_output[version_col] = version
        # Create the assortment_npi_col before selecting columns
        assortment_new_output[assortment_npi_col] = assortment_new_output[assortment_new_col]
        # Req col
        assortment_new_output = assortment_new_output[cols_required_in_output_ass_new]

        assortment_inactive_output = assortment_new_output[
            assortment_new_output[assortment_new_col] >= 1
        ]
        assortment_inactive_output[assortment_inactive_col] = np.nan
        # Req col
        assortment_inactive_output = assortment_inactive_output[
            cols_required_in_output_ass_inactive
        ]

        dates_output = concat_to_dataframe(dates_list)
        dates_output[version_col] = version
        # Req col
        dates_output = dates_output[cols_required_in_output_dates]

        npi_fcst_output = concat_to_dataframe(npi_fcst_list)
        npi_fcst_output[version_col] = version
        # Req col
        npi_fcst_output = npi_fcst_output[cols_required_in_output_npi_fcst]

    except Exception as e:
        logger.exception(
            "Exception {} for slice : {}, returning empty dataframe as output ...".format(
                e, df_keys
            )
        )
        logger.exception(e)
        return npi_fcst_output, dates_output, assortment_new_output, assortment_inactive_output

    return npi_fcst_output, dates_output, assortment_new_output, assortment_inactive_output
