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
from helpers.utils import filter_for_iteration, get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

col_mapping = {
    "Stat Fcst TL": float,
    "Stat Fcst TL CML Baseline": float,
    "Stat Fcst TL CML Residual": float,
    "Stat Fcst TL CML Holiday": float,
    "Stat Fcst TL CML Promo": float,
    "Stat Fcst TL CML Marketing": float,
    "Stat Fcst TL CML Price": float,
    "Stat Fcst TL CML Weather": float,
    "Stat Fcst TL CML External Driver": float,
    "TL Volume Loss Flag": str,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Grains,
    StatFcstFinalProfileTL,
    StatFcstL1,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    default_mapping,
    ForecastGenerationTimeBucket,
    ItemDates,
    CurrentDay,
    CMLFcstL1=pd.DataFrame(),
    MLDecompositionFlag=pd.DataFrame(),
    df_keys={},
):
    try:
        OutputList = list()
        CMLOutputList = list()
        Output_volume_loss_flag_list = list()

        for the_iteration in ForecastGenerationTimeBucket[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            the_output, cml_output, the_Output_volume_loss_flag = decorated_func(
                Grains=Grains,
                StatFcstFinalProfileTL=StatFcstFinalProfileTL,
                StatFcstL1=StatFcstL1,
                CMLFcstL1=CMLFcstL1,
                ForecastLevelData=ForecastLevelData,
                ItemMasterData=ItemMasterData,
                RegionMasterData=RegionMasterData,
                AccountMasterData=AccountMasterData,
                ChannelMasterData=ChannelMasterData,
                PnLMasterData=PnLMasterData,
                DemandDomainMasterData=DemandDomainMasterData,
                LocationMasterData=LocationMasterData,
                ItemDates=ItemDates,
                CurrentDay=CurrentDay,
                the_iteration=the_iteration,
                MLDecompositionFlag=MLDecompositionFlag,
                ForecastGenerationTimeBucket=ForecastGenerationTimeBucket,
                default_mapping=default_mapping,
                df_keys=df_keys,
            )

            OutputList.append(the_output)
            CMLOutputList.append(cml_output)
            Output_volume_loss_flag_list.append(the_Output_volume_loss_flag)

        Output = concat_to_dataframe(OutputList)
        CMLOutput = concat_to_dataframe(CMLOutputList)
        Output_volume_loss_flag = concat_to_dataframe(Output_volume_loss_flag_list)
    except Exception as e:
        logger.exception(e)
        Output = None
        CMLOutput = None
        Output_volume_loss_flag = None
    return Output, CMLOutput, Output_volume_loss_flag


def processIteration(
    Grains,
    StatFcstFinalProfileTL,
    StatFcstL1,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    ItemDates,
    CurrentDay,
    the_iteration,
    default_mapping,
    ForecastGenerationTimeBucket,
    CMLFcstL1=pd.DataFrame(),
    MLDecompositionFlag=pd.DataFrame(),
    df_keys={},
):
    plugin_name = "DP066SpreadStatFcstL1ToTL"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # configurables
    GROUP_SUM: str = "Group Sum"
    DISAGG_PROPORTION: str = "Disagg Proportion"
    CML_ITERATION_DECOMPOSITION: str = "CML Iteration Decomposition"
    ML_FCST_L1_CML_BASELINE: str = "ML Fcst L1 CML Baseline"
    ML_FCST_L1_CML_RESIDUAL: str = "ML Fcst L1 CML Residual"
    ML_FCST_L1_CML_HOLIDAY: str = "ML Fcst L1 CML Holiday"
    ML_FCST_L1_CML_PROMO: str = "ML Fcst L1 CML Promo"
    ML_FCST_L1_CML_MARKETING: str = "ML Fcst L1 CML Marketing"
    ML_FCST_L1_CML_PRICE: str = "ML Fcst L1 CML Price"
    ML_FCST_L1_CML_WEATHER: str = "ML Fcst L1 CML Weather"
    ML_FCST_L1_CML_EXTERNAL_DRIVER: str = "ML Fcst L1 CML External Driver"
    STAT_FCST_TL_CML_BASELINE: str = "Stat Fcst TL CML Baseline"
    STAT_FCST_TL_CML_RESIDUAL: str = "Stat Fcst TL CML Residual"
    STAT_FCST_TL_CML_HOLIDAY: str = "Stat Fcst TL CML Holiday"
    STAT_FCST_TL_CML_PROMO: str = "Stat Fcst TL CML Promo"
    STAT_FCST_TL_CML_MARKETING: str = "Stat Fcst TL CML Marketing"
    STAT_FCST_TL_CML_PRICE: str = "Stat Fcst TL CML Price"
    STAT_FCST_TL_CML_WEATHER: str = "Stat Fcst TL CML Weather"
    STAT_FCST_TL_CML_EXTERNAL_DRIVER: str = "Stat Fcst TL CML External Driver"
    TL_VOLUME_LOSS_FLAG: str = "TL Volume Loss Flag"
    PARTIAL_WEEK_KEY: str = "Time.[PartialWeekKey]"
    CURRENT_PARTIAL_WEEK_KEY: str = "current_partial_weekkey"
    INTRO_DATE: str = "Intro Date"
    DISCO_DATE: str = "Disco Date"
    STAT_FCST_FINAL_PROFILE_TL: str = "Stat Fcst Final Profile TL"

    # combine grains to get segmentation level
    dimensions = get_list_of_grains_from_string(input=Grains)

    cols_required_in_output = (
        [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK]
        + dimensions
        + [o9Constants.STAT_FCST_TL]
    )
    cols_required_in_cml_output = (
        [o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK]
        + dimensions
        + [
            STAT_FCST_TL_CML_BASELINE,
            STAT_FCST_TL_CML_RESIDUAL,
            STAT_FCST_TL_CML_HOLIDAY,
            STAT_FCST_TL_CML_PROMO,
            STAT_FCST_TL_CML_MARKETING,
            STAT_FCST_TL_CML_PRICE,
            STAT_FCST_TL_CML_WEATHER,
            STAT_FCST_TL_CML_EXTERNAL_DRIVER,
        ]
    )

    cols_required_in_volume_loss_flag_output = [
        o9Constants.VERSION_NAME,
        o9Constants.FORECAST_ITERATION,
        o9Constants.PLANNING_REGION,
        o9Constants.TRANSITION_ITEM,
        o9Constants.PLANNING_LOCATION,
        o9Constants.PLANNING_CHANNEL,
        o9Constants.PLANNING_PNL,
        o9Constants.TRANSITION_DEMAND_DOMAIN,
        o9Constants.PLANNING_ACCOUNT,
        TL_VOLUME_LOSS_FLAG,
    ]
    Output = pd.DataFrame(columns=cols_required_in_output)
    Output_volume_loss_flag = pd.DataFrame(columns=cols_required_in_volume_loss_flag_output)
    CMLOutput = pd.DataFrame(columns=cols_required_in_cml_output)
    try:
        if StatFcstFinalProfileTL.empty:
            logger.warning(f"StatFcstFinalProfileTL is empty for {df_keys}")
            return Output, CMLOutput, Output_volume_loss_flag

        if StatFcstL1.empty:
            logger.warning(f"StatFcstL1 is empty for {df_keys}")
            return Output, CMLOutput, Output_volume_loss_flag

        if ForecastLevelData.empty:
            logger.warning(f"ForecastLevelData is empty for {df_keys}")
            return Output, CMLOutput, Output_volume_loss_flag

        forecastgenbucketname = ForecastGenerationTimeBucket[
            o9Constants.FORECAST_GEN_TIME_BUCKET
        ].values[0]
        master_data_dict = {}
        master_data_dict["Item"] = ItemMasterData
        master_data_dict["Channel"] = ChannelMasterData
        master_data_dict["Demand Domain"] = DemandDomainMasterData
        master_data_dict["Region"] = RegionMasterData
        master_data_dict["Account"] = AccountMasterData
        master_data_dict["PnL"] = PnLMasterData
        master_data_dict["Location"] = LocationMasterData

        default_cols = []
        for dim in default_mapping:
            cols = [x for x in StatFcstFinalProfileTL.columns if dim in x]
            for col in cols:
                if (
                    len(StatFcstFinalProfileTL[[col]].drop_duplicates()) == 1
                    and StatFcstFinalProfileTL[col].values[0] == default_mapping[dim]
                ):
                    # filtering master dataframe intersections for which the value is the defaults
                    master_data_dict[dim] = master_data_dict[dim][
                        master_data_dict[dim][col] == default_mapping[dim]
                    ]
                    default_cols += [col]

        level_cols = [x for x in ForecastLevelData.columns if "Level" in x]

        all_stat_grains = list()
        for the_col in level_cols:

            # extract 'Item' from 'Item Level'
            the_dim = the_col.split(" Level")[0]

            logger.debug(f"the_dim : {the_dim}")

            # transition/planning item
            if the_dim in ["Item", "Demand Domain"]:
                the_relevant_col = the_dim + ".[Transition " + the_dim + "]"
            else:
                the_relevant_col = the_dim + ".[Planning " + the_dim + "]"

            logger.debug(f"the_relevant_col : {the_relevant_col}")

            if the_relevant_col not in default_cols:
                # Item.[Stat Item]
                the_stat_col = the_dim + ".[Stat " + the_dim + "]"
                logger.debug(f"the_stat_col : {the_stat_col}")
                all_stat_grains.append(the_stat_col)

                the_dim_data = master_data_dict[the_dim]

                # Eg. the_level = Item.[L2]
                the_level = the_dim + ".[" + ForecastLevelData[the_col].iloc[0] + "]"
                logger.debug(f"the_level : {the_level}")

                # copy values from L2 to Stat Item
                the_dim_data[the_stat_col] = the_dim_data[the_level]

                # identify the relevant stat members for the iteration
                relevant_stat_members_for_the_iteration = list(StatFcstL1[the_stat_col].unique())

                if not relevant_stat_members_for_the_iteration:
                    logger.warning(f"No data found for {the_stat_col} in StatFcstL1")
                    continue
                else:
                    logger.debug(
                        f"Filtering {the_dim} master for {len(relevant_stat_members_for_the_iteration)} {the_stat_col} members"
                    )

                    # filter dim data for relevant stat members
                    filter_clause = the_dim_data[the_stat_col].isin(
                        relevant_stat_members_for_the_iteration
                    )
                    the_dim_data = the_dim_data[filter_clause]

                    if the_dim_data.empty:
                        logger.warning(
                            f"No {the_dim} found after filtering for relevant_stat_members_for_the_iteration"
                        )
                        continue

                # select only relevant columns
                the_dim_data = the_dim_data[[the_relevant_col, the_stat_col]].drop_duplicates()

                # Newly Added Parent Condition - Check Erratic Data condition only if Stat != Planning Level for Item & Demand Domain
                if the_level not in [
                    o9Constants.PLANNING_ITEM,
                    o9Constants.PLANNING_DEMAND_DOMAIN,
                ]:
                    if the_dim_data[the_relevant_col].nunique() != len(the_dim_data):
                        duplicates = the_dim_data[the_dim_data[the_relevant_col].duplicated()]
                        logger.warning(
                            f"Erratic Master data:\n{duplicates.head().to_csv(index=False)}"
                        )
                        logger.warning(
                            "Will continue with the clean data, but check the master data for duplicates"
                        )
                        # drop duplicated values
                        the_dim_data = the_dim_data[~the_dim_data[the_relevant_col].duplicated()]

                # join with Final Profile TL
                StatFcstFinalProfileTL = StatFcstFinalProfileTL.merge(
                    the_dim_data, on=the_relevant_col, how="inner"
                )

                logger.debug("-------------------------")

        ML_decomposition_flag = MLDecompositionFlag[CML_ITERATION_DECOMPOSITION].values[0]
        if ML_decomposition_flag:
            if len(CMLFcstL1) > 0:
                StatFcstL1 = StatFcstL1.merge(
                    CMLFcstL1,
                    on=[o9Constants.VERSION_NAME] + all_stat_grains + [o9Constants.PARTIAL_WEEK],
                    how="left",
                )
            else:
                ML_decomposition_flag = False
                logger.warning(
                    "CML Fcst L1 empty for this iteration, returning empty CML outputs ..."
                )

        df = StatFcstFinalProfileTL.copy()
        # join with stat fcst l1
        df = df.merge(
            StatFcstL1,
            on=[o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + all_stat_grains,
            how="outer",
            indicator=True,
        )
        df = df.drop(
            columns=[
                o9Constants.TRANSITION_ITEM,
                o9Constants.TRANSITION_DEMAND_DOMAIN,
                o9Constants.PLANNING_REGION,
                o9Constants.PLANNING_LOCATION,
                o9Constants.PLANNING_CHANNEL,
                o9Constants.PLANNING_PNL,
                o9Constants.PLANNING_ACCOUNT,
            ]
        )

        # Join with ItemDates to get Intro and Disco Date populated
        df = df.merge(ItemDates, on=[o9Constants.VERSION_NAME] + all_stat_grains, how="left")

        df = df[df[STAT_FCST_FINAL_PROFILE_TL].isnull() | (df[STAT_FCST_FINAL_PROFILE_TL] == 0)]

        df[CURRENT_PARTIAL_WEEK_KEY] = CurrentDay[PARTIAL_WEEK_KEY].iloc[0]
        df[PARTIAL_WEEK_KEY] = pd.to_datetime(df[PARTIAL_WEEK_KEY])
        df[INTRO_DATE] = pd.to_datetime(df[INTRO_DATE])
        df[DISCO_DATE] = pd.to_datetime(df[DISCO_DATE])
        df[CURRENT_PARTIAL_WEEK_KEY] = pd.to_datetime(df[CURRENT_PARTIAL_WEEK_KEY])

        # Initialize the result
        max_forecast_date = pd.to_datetime(StatFcstL1["Time.[PartialWeekKey]"]).max()
        max_forecast_date = max_forecast_date.tz_localize(None)

        def evaluate_flags(group):
            current_date = group[CURRENT_PARTIAL_WEEK_KEY].iloc[0]

            # Strip timezones from all datetime columns in group
            group[DISCO_DATE] = pd.to_datetime(group[DISCO_DATE]).dt.tz_localize(None)
            group[INTRO_DATE] = pd.to_datetime(group[INTRO_DATE]).dt.tz_localize(None)

            # Ensure current_date and max_forecast_date are also naive
            current_date = pd.to_datetime(group[CURRENT_PARTIAL_WEEK_KEY].iloc[0]).tz_localize(None)
            max_forecast_date = pd.to_datetime("2025-12-27")

            # 1. Intro Date in the Past with no Actuals
            any_disco_in_range = (
                group[DISCO_DATE].dropna().between(current_date, max_forecast_date).any()
                if group[DISCO_DATE].notna().any()
                else False
            )
            others_intro_before_now = (
                (group[INTRO_DATE].dropna() < current_date).all()
                if group[INTRO_DATE].notna().any()
                else False
            )

            if any_disco_in_range and others_intro_before_now:
                group[TL_VOLUME_LOSS_FLAG] = "Intro Date in the Past with no Actuals"
                return group

            # 2. All discontinued items
            all_disco_in_range = group[DISCO_DATE].between(
                current_date, max_forecast_date
            ).all() and (
                group[INTRO_DATE].isna().all()  # No intro dates at all
                or (group[INTRO_DATE] <= group[DISCO_DATE]).all()  # All intro <= disco
            )

            if all_disco_in_range:
                group[TL_VOLUME_LOSS_FLAG] = "All discontinued Items"
                return group

            # Only keep one row per Transition Item to check intro/disco gap — take max intro and min disco
            transition_summary = (
                group.groupby("Item.[Transition Item]")[[INTRO_DATE, DISCO_DATE]]
                .agg({INTRO_DATE: "min", DISCO_DATE: "max"})
                .sort_values(by=INTRO_DATE)
                .reset_index()
            )

            if len(transition_summary) == 1:
                intro_date = pd.to_datetime(group[INTRO_DATE].max())
                disco_date = pd.to_datetime(group[DISCO_DATE].min())
                if pd.notna(intro_date) and pd.notna(disco_date):
                    if intro_date >= disco_date:
                        group[TL_VOLUME_LOSS_FLAG] = "Transition Period Gap"
                        return group

            else:
                # Define threshold based on forecast bucket
                if forecastgenbucketname.lower() == "week":
                    threshold = pd.Timedelta(weeks=1)
                elif forecastgenbucketname.lower() == "month":
                    threshold = pd.Timedelta(days=30)
                elif forecastgenbucketname.lower() == "planning month":
                    threshold = pd.Timedelta(days=30)
                elif forecastgenbucketname.lower() == "quarter":
                    threshold = pd.Timedelta(days=90)
                elif forecastgenbucketname.lower() == "planning quarter":
                    threshold = pd.Timedelta(days=90)
                else:
                    threshold = pd.Timedelta(0)

                # Compare disco of current with intro of next
                for i in range(len(transition_summary) - 1):
                    curr_disco = transition_summary.loc[i, DISCO_DATE]
                    next_intro = transition_summary.loc[i + 1, INTRO_DATE]

                    if pd.isna(curr_disco):
                        curr_disco = pd.Timestamp("2099-12-31")
                    if pd.isna(next_intro):
                        next_intro = pd.Timestamp("2000-01-01")

                    if pd.notna(curr_disco) and pd.notna(next_intro):
                        if isinstance(threshold, pd.Timedelta):
                            if abs(next_intro - curr_disco) > threshold:
                                group[TL_VOLUME_LOSS_FLAG] = "Transition Period Gap"
                                break
                        else:
                            if abs((next_intro - curr_disco).days) > threshold:
                                group[TL_VOLUME_LOSS_FLAG] = "Transition Period Gap"
                                break

            # 4. Zero in profile (aggregated)
            profile_sum = group[STAT_FCST_FINAL_PROFILE_TL].sum(skipna=True)
            if profile_sum == 0:
                group[TL_VOLUME_LOSS_FLAG] = "Zero in profile"
                return group

            # 5. Unknown (already set by default)
            group[TL_VOLUME_LOSS_FLAG] = "Unknown"
            return group

        # Apply for each Stat Item
        grouped = df.groupby(o9Constants.STAT_ITEM)

        results = []  # to store processed groups

        for stat_item, group in grouped:
            processed_group = evaluate_flags(group)
            results.append(processed_group)

        if results:
            # Concatenate all the processed groups
            final_df = pd.concat(results, ignore_index=True)
            final_df.reset_index(drop=True, inplace=True)
            final_df[o9Constants.FORECAST_ITERATION] = the_iteration
            Output_volume_loss_flag = final_df[cols_required_in_volume_loss_flag_output]
            Output_volume_loss_flag = Output_volume_loss_flag.drop_duplicates()
            Output_volume_loss_flag = Output_volume_loss_flag.dropna(
                subset=[o9Constants.TRANSITION_ITEM]
            )

        else:
            logger.warning(f"No volume loss for {df_keys}")
            Output_volume_loss_flag = pd.DataFrame(columns=cols_required_in_volume_loss_flag_output)

        logger.info("TL VOLUME LOSS FLAG calculation done ...")

        # join with stat fcst l1 - to retain relevant intersections
        StatFcstFinalProfileTL = StatFcstFinalProfileTL.merge(
            StatFcstL1,
            on=[o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + all_stat_grains,
            how="inner",
        )

        # create group sum - at stat level
        StatFcstFinalProfileTL[GROUP_SUM] = StatFcstFinalProfileTL.groupby(
            all_stat_grains + [o9Constants.PARTIAL_WEEK]
        )[o9Constants.STAT_FCST_FINAL_PROFILE_TL].transform("sum")

        # create proportions
        StatFcstFinalProfileTL[DISAGG_PROPORTION] = (
            StatFcstFinalProfileTL[o9Constants.STAT_FCST_FINAL_PROFILE_TL]
            / StatFcstFinalProfileTL[GROUP_SUM]
        )

        # multiply value with proportion
        StatFcstFinalProfileTL[o9Constants.STAT_FCST_TL] = (
            StatFcstFinalProfileTL[o9Constants.STAT_FCST_L1]
            * StatFcstFinalProfileTL[DISAGG_PROPORTION]
        )

        # select relevant columns
        Output = StatFcstFinalProfileTL[cols_required_in_output].drop_duplicates()
        # Newly Added - Agg across Transition Item & Transition Demand Domain columns
        Output = (
            Output.groupby([o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + dimensions)
            .agg({o9Constants.STAT_FCST_TL: sum})
            .reset_index()
        )
        # calculating Fcst TL for all CML drivers, Stat Fcst TL = ML Fcst L1 * Disagg Proportion
        if ML_decomposition_flag:
            fields = [
                (STAT_FCST_TL_CML_BASELINE, ML_FCST_L1_CML_BASELINE),
                (STAT_FCST_TL_CML_RESIDUAL, ML_FCST_L1_CML_RESIDUAL),
                (STAT_FCST_TL_CML_HOLIDAY, ML_FCST_L1_CML_HOLIDAY),
                (STAT_FCST_TL_CML_PROMO, ML_FCST_L1_CML_PROMO),
                (STAT_FCST_TL_CML_MARKETING, ML_FCST_L1_CML_MARKETING),
                (STAT_FCST_TL_CML_PRICE, ML_FCST_L1_CML_PRICE),
                (STAT_FCST_TL_CML_WEATHER, ML_FCST_L1_CML_WEATHER),
                (
                    STAT_FCST_TL_CML_EXTERNAL_DRIVER,
                    ML_FCST_L1_CML_EXTERNAL_DRIVER,
                ),
            ]

            for stat_field, ml_field in fields:
                StatFcstFinalProfileTL[stat_field] = (
                    StatFcstFinalProfileTL[ml_field] * StatFcstFinalProfileTL[DISAGG_PROPORTION]
                )
            CMLOutput = StatFcstFinalProfileTL[cols_required_in_cml_output].drop_duplicates()
            CMLOutput = (
                CMLOutput.groupby([o9Constants.VERSION_NAME, o9Constants.PARTIAL_WEEK] + dimensions)
                .agg(
                    {
                        STAT_FCST_TL_CML_BASELINE: sum,
                        STAT_FCST_TL_CML_RESIDUAL: sum,
                        STAT_FCST_TL_CML_HOLIDAY: sum,
                        STAT_FCST_TL_CML_PROMO: sum,
                        STAT_FCST_TL_CML_MARKETING: sum,
                        STAT_FCST_TL_CML_PRICE: sum,
                        STAT_FCST_TL_CML_WEATHER: sum,
                        STAT_FCST_TL_CML_EXTERNAL_DRIVER: sum,
                    }
                )
                .reset_index()
            )
            CMLOutput = StatFcstFinalProfileTL[cols_required_in_cml_output]
        Output = StatFcstFinalProfileTL[cols_required_in_output]
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)
        CMLOutput = pd.DataFrame(columns=cols_required_in_cml_output)
        Output_volume_loss_flag = pd.DataFrame(columns=cols_required_in_volume_loss_flag_output)
    return Output, CMLOutput, Output_volume_loss_flag
