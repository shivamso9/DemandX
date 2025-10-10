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


col_mapping = {"Fcst Outside PLC Horizon Flag": float}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    ConsensusFcst,
    IntroDiscDates,
    TimeDimension,
    CurrentDay,
    Grains,
    OutputMeasure,
    df_keys,
):
    plugin_name = "DP060PopulateFcstOutsidePLCHorizonFlag"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

    # Configurables
    version_col = "Version.[Version Name]"
    partial_week_col = "Time.[Partial Week]"
    partial_week_key_col = "Time.[PartialWeekKey]"
    intro_date_col = "Intro Date"
    disco_date_col = "Disco Date"
    day_key_col = "Time.[DayKey]"
    intro_time_bucket_col = "Intro Time Bucket"
    disco_time_bucket_col = "Disco Time Bucket"

    # split on delimiter and obtain grains
    all_grains = Grains.split(",")

    # remove leading/trailing spaces if any
    all_grains = [x.strip() for x in all_grains]

    # combine grains to get forecast level
    profile_level = [str(x) for x in all_grains if x != "NA" and x != ""]

    cols_required_in_output = [version_col] + profile_level + [partial_week_col, OutputMeasure]
    Output = pd.DataFrame(columns=cols_required_in_output)
    try:
        if ConsensusFcst.empty:
            logger.warning("ConsensusFcst is empty, returning without further execution")
            return Output

        if IntroDiscDates.empty:
            logger.warning("IntroDiscDates is empty, returning without further execution")
            return Output

        if CurrentDay.empty:
            logger.warning("CurrentDay is empty, returning without further execution")
            return Output

        relevant_time_key_col = partial_week_key_col
        relevant_time_cols = [day_key_col, relevant_time_key_col]

        day_mapping = TimeDimension[relevant_time_cols].drop_duplicates()

        pw_mapping = TimeDimension[[partial_week_col, partial_week_key_col]].drop_duplicates()

        # join to get pw key
        ConsensusFcst = ConsensusFcst.merge(pw_mapping, on=partial_week_col, how="inner")

        current_day_key = CurrentDay[day_key_col].iloc[0]
        logger.debug(f"current_day_key : {current_day_key}")
        last_pw_key = ConsensusFcst[partial_week_key_col].max()
        logger.debug(f"last_pw_key : {last_pw_key}")

        IntroDiscDates[intro_date_col].fillna(current_day_key, inplace=True)
        IntroDiscDates[disco_date_col].fillna(last_pw_key, inplace=True)

        IntroDiscDates[intro_date_col] = pd.to_datetime(IntroDiscDates[intro_date_col])

        IntroDiscDates[disco_date_col] = pd.to_datetime(IntroDiscDates[disco_date_col])

        # removing timestamps from Intro and Disco Dates if present
        IntroDiscDates[intro_date_col] = IntroDiscDates[intro_date_col].dt.normalize()

        IntroDiscDates[disco_date_col] = IntroDiscDates[disco_date_col].dt.normalize()

        logger.debug("Joining intro/disc dates with time mapping to get relevant time buckets ...")
        # Join intro/disc dates with day mapping to get the time bucket corresponding to Intro/Disc Date
        IntroDiscDates = IntroDiscDates.merge(
            day_mapping.rename(columns={day_key_col: intro_date_col}),
            on=intro_date_col,
            how="inner",
        )
        IntroDiscDates.rename(
            columns={relevant_time_key_col: intro_time_bucket_col},
            inplace=True,
        )

        IntroDiscDates = IntroDiscDates.merge(
            day_mapping.rename(columns={day_key_col: disco_date_col}),
            on=disco_date_col,
            how="inner",
        )
        IntroDiscDates.rename(
            columns={relevant_time_key_col: disco_time_bucket_col},
            inplace=True,
        )

        logger.debug("Joining ConsensusFcst with IntroDiscDates ...")

        # join ConsensusFcst with IntroDiscDates
        ConsensusFcst = ConsensusFcst.merge(
            IntroDiscDates, on=[version_col] + profile_level, how="inner"
        )

        if ConsensusFcst.empty:
            logger.warning(
                "No common records b/w ConsensusFcst and IntroDiscDates, please check input data ..."
            )
            return Output

        logger.debug("Applying filter ...")

        # filter records outside the PLC Horizon
        plc_filter_clause = (
            ConsensusFcst[relevant_time_key_col] < ConsensusFcst[intro_time_bucket_col]
        ) | (ConsensusFcst[relevant_time_key_col] > ConsensusFcst[disco_time_bucket_col])
        ConsensusFcst = ConsensusFcst[plc_filter_clause]

        logger.debug(f"Adding {OutputMeasure} with value 1.0 to filtered records ...")
        # apply flag
        ConsensusFcst[OutputMeasure] = 1.0

        # filter required columns
        Output = ConsensusFcst[cols_required_in_output]

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
        Output = pd.DataFrame(columns=cols_required_in_output)

    return Output
