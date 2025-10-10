import logging
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from o9Reference.common_utils.dataframe_utils import create_cartesian_product
from o9Reference.common_utils.decorators import (
    convert_category_cols_to_str,
    map_output_columns_to_dtypes,
)
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed

from helpers.o9Constants import o9Constants

logger = logging.getLogger("o9_logger")

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3
pd.options.mode.chained_assignment = None

# col_mapping = {
#     "DA - 1 Int": float,
#     "DA - 2 Int": float,
#     "DA - 3 Int": float,
#     "DA - 4 Int": float,
#     "DA - 5 Int": float,
#     "DA - 6 Int": float,
#     "Planner Input Assortment": float,
# }

col_mapping = {
    "Planner Input Assortment": float,
}


def get_DA_Int_for_each_initiative(
    group,
    current_time_key,
    levels_for_check,
    initiative_type_col,
    consensus_fcst,
    consensus_assort,
    planner_input,
    TimeDimension,
    UOMConversion,
    cols_required_in_DA_output,
    uom_conversion,
):
    DISAGGREGATION_BASIS = "DA Disaggregation Basis"
    ASSORTMENT_BASIS = "DA Assortment Basis"
    is_already_touchless = "Is Already Touchless"
    try:
        group.reset_index(drop=True, inplace=True)
        initiative = group[o9Constants.INITIATIVE].values[0]
        disagg_basis = group[DISAGGREGATION_BASIS].values[0]
        assort_basis = group[ASSORTMENT_BASIS].values[0]

        UOMConversion = UOMConversion.merge(
            group["DA UOM"].drop_duplicates(),
            left_on="UOM.[UOM]",
            right_on="DA UOM",
            how="inner",
        )

        target_dims = [
            col
            for col in cols_required_in_DA_output
            if (("[" in col) and (col not in [o9Constants.PARTIAL_WEEK, o9Constants.INITIATIVE]))
        ]
        # Baseline Fcst Inclusion
        if disagg_basis == "Stat Fcst":
            disagg_basis = "Stat Fcst L0"
        elif disagg_basis == "Baseline Fcst":
            disagg_basis = "Consensus Baseline Fcst"
        if disagg_basis not in consensus_fcst:
            logger.warning(
                f"Disaggregation Basis {disagg_basis} column not found in Candidate Fcst, returning empty output for initiative {initiative}"
            )
            return pd.DataFrame(columns=cols_required_in_DA_output)
        if assort_basis not in consensus_assort:
            logger.warning(
                f"Assortment Basis {assort_basis} column not found in Candidate Assortment, returning empty output for initiative {initiative}"
            )
            return pd.DataFrame(columns=cols_required_in_DA_output)

        # taking all dimensions from assortment df so that we can focus on assort basis and dropnas effectively
        dims_in_assort = [col for col in consensus_assort if "[" in col]
        consensus_assort = consensus_assort[
            dims_in_assort + [assort_basis, is_already_touchless]
        ].dropna(subset=[assort_basis])

        consensus_assort = consensus_assort[consensus_assort[assort_basis].astype(int) == 1]

        # taking all dimensions from candidate fcst df so that we can focus on disagg basis and dropnas effectively
        dims_in_fcst = [col for col in consensus_fcst if "[" in col]
        consensus_fcst = consensus_fcst[dims_in_fcst + [disagg_basis]].drop_duplicates()

        planner_input = planner_input[
            planner_input[o9Constants.INITIATIVE] == group[o9Constants.INITIATIVE].values[0]
        ].drop_duplicates()
        time_level = "Time.[" + group["DA Time Level"].values[0] + "]"
        time_level_key = "Time.[" + group["DA Time Level"].values[0].replace(" ", "") + "Key]"
        initiative_type = group[initiative_type_col].values[0]
        current_time_key = pd.to_datetime(current_time_key)
        planner_input = planner_input[planner_input[time_level_key] > current_time_key]

        if time_level not in planner_input.columns:
            logger.warning(
                f"{time_level} not found in Planner Input Time levels, returning empty output for the initiative {group[o9Constants.INITIATIVE]}..."
            )
            return pd.DataFrame(columns=cols_required_in_DA_output)
        else:
            planner_input_grouped = planner_input.groupby(
                [o9Constants.VERSION_NAME, o9Constants.INITIATIVE, time_level],
                as_index=False,
            ).agg({"Planner Input": "sum"})
        initiative_df = pd.DataFrame()
        alternate_item_scope_flag = False
        for level in levels_for_check:
            # Constructing the column names dynamically
            level_column = f"DA {level} Level"
            scope_column = f"DA {level} Scope"
            if level == "Alternate Item":
                alternate_item_scope = group[scope_column].values[0].split(",")
                alternate_item_scope_flag = True
                alternate_item_scope = pd.DataFrame(
                    {"Item.[" + str(group[level_column].values[0]) + "]": alternate_item_scope}
                )
            else:
                initiative_df[level + ".[" + str(group[level_column].values[0]) + "]"] = group[
                    scope_column
                ]

        initiative_df = pd.DataFrame(initiative_df)

        if alternate_item_scope_flag:
            initiative_df = create_cartesian_product(initiative_df, alternate_item_scope)
        missing_cols = initiative_df.columns.difference(set(consensus_assort.columns))
        relevant_columns = initiative_df.columns.to_list()

        if len(missing_cols) > 0:
            logger.warning(
                f"{missing_cols} columns missing in Consensus inputs, returning empty output for the initiative {group[o9Constants.INITIATIVE].values[0]}..."
            )
            return pd.DataFrame(columns=cols_required_in_DA_output)

        else:
            group_columns = [
                col for col in relevant_columns if col not in alternate_item_scope.columns
            ] + [
                o9Constants.VERSION_NAME,
                time_level,
            ]

            group["DA Start Date"] = pd.to_datetime(
                group["DA Start Date"], utc=True
            ).dt.tz_localize(None)
            group["DA End Date"] = pd.to_datetime(group["DA End Date"], utc=True).dt.tz_localize(
                None
            )

            TimeDimension = TimeDimension[
                (
                    TimeDimension[o9Constants.DAY_KEY]
                    >= pd.to_datetime(group["DA Start Date"].values[0])
                )
                & (
                    TimeDimension[o9Constants.DAY_KEY]
                    <= pd.to_datetime(group["DA End Date"].values[0])
                )
            ]
            if time_level == o9Constants.PARTIAL_WEEK:
                relevant_time_df = TimeDimension[o9Constants.PARTIAL_WEEK].drop_duplicates()
            else:
                relevant_time_df = TimeDimension[
                    [o9Constants.PARTIAL_WEEK, time_level]
                ].drop_duplicates()

            consensus_assort = consensus_assort.merge(
                relevant_time_df,
                how="cross",
            )

            # filter intersections for which consensus assortment is 1
            consensus_fcst = consensus_assort.merge(consensus_fcst, how="left")

            # filter the intersections which are under the initiative's scope
            consensus_fcst = consensus_fcst.merge(initiative_df, on=relevant_columns, how="inner")

            if time_level == o9Constants.PARTIAL_WEEK:
                # If time_level is PARTIAL_WEEK, only keep partial week
                cols_req = list(set(relevant_columns + target_dims)) + [
                    o9Constants.PARTIAL_WEEK,
                    disagg_basis,
                    assort_basis,
                    is_already_touchless,
                ]
            else:
                # If time_level is not PARTIAL_WEEK, keep both time level and PW columns
                cols_req = list(set(relevant_columns + target_dims)) + [
                    o9Constants.PARTIAL_WEEK,
                    time_level,
                    disagg_basis,
                    assort_basis,
                    is_already_touchless,
                ]
            consensus_fcst = consensus_fcst[cols_req].drop_duplicates()
            # if len(consensus_fcst[disagg_basis].unique()) > 1:
            # consensus_fcst = consensus_fcst.dropna(subset=[disagg_basis])

            # inner join to ensure we have UOMConversion values for all Initiatives
            consensus_fcst = consensus_fcst.merge(
                UOMConversion,
                how="left",
            )
            consensus_fcst[uom_conversion] = consensus_fcst[uom_conversion].fillna(1.0)
            if consensus_fcst.empty:
                logger.warning("No uom value for the relevant intersections ...")
                logger.warning("Returning empty dataframe ...")
                return pd.DataFrame(columns=cols_required_in_DA_output)

            consensus_fcst = consensus_fcst.merge(
                planner_input_grouped[[time_level, "Planner Input"]],
                on=time_level,
                how="left",
            )
            # if consensus_fcst.empty:
            #     logger.warning("No planner input value for the relevant intersections ...")
            #     logger.warning("Returning empty dataframe ...")
            #     return pd.DataFrame(columns=cols_required_in_DA_output)

            consensus_fcst[disagg_basis] = (
                consensus_fcst[disagg_basis] * consensus_fcst[uom_conversion]
            )
            consensus_fcst[o9Constants.PLANNER_INPUT_ASSORTMENT] = np.where(
                ((consensus_fcst[disagg_basis] != 0) & (consensus_fcst[disagg_basis].notna())),
                consensus_fcst[disagg_basis],
                consensus_fcst[assort_basis],
            )
            consensus_fcst_touchless = consensus_fcst[consensus_fcst[is_already_touchless] == "Y"]
            consensus_fcst = consensus_fcst[consensus_fcst[is_already_touchless] != "Y"]
            consensus_fcst["Candidate Fcst Sum Intersection"] = consensus_fcst.groupby(
                group_columns
            )[disagg_basis].transform("sum")
            consensus_fcst["Group Length"] = consensus_fcst.groupby(group_columns)[
                disagg_basis
            ].transform("size")

            consensus_fcst["mask"] = np.where(
                (
                    (consensus_fcst["Candidate Fcst Sum Intersection"].isna())
                    | (consensus_fcst["Candidate Fcst Sum Intersection"] == 0)
                ),
                True,
                False,
            )

            # Calculate the 'initiative_type + " Int"' based on the condition
            consensus_fcst[initiative_type + " Int"] = np.where(
                consensus_fcst["mask"],
                consensus_fcst["Planner Input"]
                / consensus_fcst[uom_conversion]
                / consensus_fcst["Group Length"],  # If all 'Candidate Fcst' are NaN or 0
                consensus_fcst[disagg_basis]
                * consensus_fcst["Planner Input"]
                / consensus_fcst[uom_conversion]
                / consensus_fcst["Candidate Fcst Sum Intersection"].replace(0, np.nan),  # Otherwise
            )

            consensus_fcst[disagg_basis] = (
                consensus_fcst[disagg_basis] / consensus_fcst[uom_conversion]
            )
            consensus_fcst[initiative_type + " Int"] = consensus_fcst[
                initiative_type + " Int"
            ].replace(0, np.nan)

            # Copy the assortment values to disagg basis, in case of equal split - to populate Planner Input assortment
            consensus_fcst[disagg_basis] = np.where(
                consensus_fcst["mask"],
                consensus_fcst[assort_basis],  # If all 'Candidate Fcst' are NaN or 0
                consensus_fcst[disagg_basis],  # Otherwise
            )
            consensus_fcst = pd.concat([consensus_fcst, consensus_fcst_touchless])
            consensus_fcst[o9Constants.INITIATIVE] = group[o9Constants.INITIATIVE].values[0]
            # consensus_fcst.rename(
            #     columns={disagg_basis: o9Constants.PLANNER_INPUT_ASSORTMENT},
            #     inplace=True,
            # )
    except Exception as e:
        logger.warning(e)
        return pd.DataFrame(columns=cols_required_in_DA_output)
    return consensus_fcst


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    InitiativeInput: pd.DataFrame,
    PlannerInput: pd.DataFrame,
    CandidateAssortment: pd.DataFrame,
    CandidateFcst: pd.DataFrame,
    TimeDimension: pd.DataFrame,
    CurrentDay: pd.DataFrame,
    Grains: str = "",
    UOMConversion: Optional[pd.DataFrame] = None,
    df_keys: dict = {},
):
    plugin_name = "DP091DARespread"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    initiative_type_col = "Initiative.[Initiative Type]"
    uom_col = "UOM.[UOM]"
    uom_conversion = "UOM Conversion"

    if len(Grains) == 0:
        logger.warning("Grains cannot be empty ...")
        logger.warning("Returning empty dataframe ...")
        return pd.DataFrame()

    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    output_level = [str(x) for x in all_grains if x != "NA" and x != ""]
    dimensions = [x.split(".")[0] for x in output_level]

    try:
        # assert (
        #     PlannerInput is not None and len(PlannerInput) > 0
        # ), "Planner Input empty, returning empty outputs ..."
        assert (
            InitiativeInput is not None and len(InitiativeInput) > 0
        ), "Initiative Input empty, returning empty outputs ..."

        InitiativeTypes = InitiativeInput[initiative_type_col]
        Type_df = InitiativeTypes.to_frame()
        Type_df["Number"] = Type_df[initiative_type_col].str.extract(r"(\d+)").astype(int)
        TypeMax = Type_df["Number"].max()

        cols_required_in_DA_output = (
            [
                o9Constants.VERSION_NAME,
                o9Constants.INITIATIVE,
                o9Constants.PARTIAL_WEEK,
            ]
            + output_level
            + ["DA - " + str(i) + " Int" for i in range(1, TypeMax + 1)]
            + [o9Constants.PLANNER_INPUT_ASSORTMENT]
        )

        DAIntOutput = pd.DataFrame(columns=cols_required_in_DA_output)
        for initiative_type in range(1, TypeMax + 1):
            col_mapping["DA - " + str(initiative_type) + " Int"] = "float"

        current_time_key = CurrentDay["Time.[DayKey]"].iloc[0]

        if UOMConversion is None:
            UOMConversion = pd.DataFrame(
                columns=[
                    o9Constants.VERSION_NAME,
                    o9Constants.PLANNING_ITEM,
                    uom_col,
                    uom_conversion,
                ]
            )
        if UOMConversion.empty:
            UOMConversion = pd.DataFrame(
                columns=[
                    o9Constants.VERSION_NAME,
                    o9Constants.PLANNING_ITEM,
                    uom_col,
                    uom_conversion,
                ]
            )

        initiatives_req = PlannerInput[o9Constants.INITIATIVE].unique()
        logger.info(f"Running plugin for the initiatives{initiatives_req}...")

        # InitiativeInput = InitiativeInput[
        #     InitiativeInput[o9Constants.INITIATIVE].isin(initiatives_req)
        # ]
        CandidateAssortment = CandidateAssortment[CandidateAssortment["Assortment Final"] == 1]

        if len(InitiativeInput) == 0:
            logger.warning(
                "Initiative input not present after filtering for required initiatives ...\nReturning empty dataframes"
            )
            return DAIntOutput

        levels_for_check = dimensions + ["Alternate Item"]

        time_mapping = TimeDimension[
            [
                o9Constants.PARTIAL_WEEK,
                o9Constants.PARTIAL_WEEK_KEY,
                o9Constants.WEEK,
                o9Constants.WEEK_KEY,
                o9Constants.MONTH,
                o9Constants.MONTH_KEY,
                o9Constants.PLANNING_MONTH,
                o9Constants.PLANNING_MONTH_KEY,
            ]
        ]

        PlannerInput = PlannerInput.merge(time_mapping, on=o9Constants.PARTIAL_WEEK, how="inner")

        TimeDimension[o9Constants.DAY_KEY] = TimeDimension[o9Constants.DAY_KEY].dt.tz_localize(None)

        # InitiativeInput = InitiativeInput[
        #     ~InitiativeInput[da_uom].isna()
        # ]

        # if InitiativeInput.empty:
        #     logger.warning(
        #         f"UOM conversion values not present for any initiative...\nReturning empty dataframes"
        #     )
        #     return DAIntOutput

        all_results = Parallel(n_jobs=4, verbose=1)(
            delayed(get_DA_Int_for_each_initiative)(
                group=group,
                current_time_key=current_time_key,
                levels_for_check=levels_for_check,
                initiative_type_col=initiative_type_col,
                consensus_fcst=CandidateFcst,
                consensus_assort=CandidateAssortment,
                planner_input=PlannerInput,
                TimeDimension=TimeDimension,
                UOMConversion=UOMConversion,
                cols_required_in_DA_output=cols_required_in_DA_output,
                uom_conversion=uom_conversion,
            )
            for name, group in InitiativeInput.groupby(o9Constants.INITIATIVE)
        )
        DAIntOutput = pd.concat([DAIntOutput] + all_results, ignore_index=True)
        DAIntOutput = DAIntOutput[cols_required_in_DA_output]

        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        DAIntOutput = pd.DataFrame(columns=cols_required_in_DA_output)
    return DAIntOutput
