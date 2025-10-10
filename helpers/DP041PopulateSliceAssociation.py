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
from helpers.utils import (
    add_dim_suffix,
    filter_for_iteration,
    get_list_of_grains_from_string,
)

logger = logging.getLogger("o9_logger")
pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3


def get_sequence(
    intersections: pd.DataFrame,
    num_intersections_in_one_slice: int,
):
    sequence_col = "Sequence.[Sequence]"
    index_col = "index"
    Totalintersections = len(intersections)
    if Totalintersections < num_intersections_in_one_slice:
        NumBuckets = 1
    else:
        # determine the number of slice buckets by diving (whole number division) total num of intersections by NumIntersectionsInOneSlice
        # add 1 to ensure that there's atleast one bucket
        NumBuckets = (Totalintersections // num_intersections_in_one_slice) + 1

    logger.info("NumBuckets : {}".format(NumBuckets))

    # reset index to have continous indexes from 0 to n
    intersections.reset_index(inplace=True)

    if NumBuckets > 1:
        # take modulus of row number by NumBuckets to assign each row to a different slice sequence
        intersections[sequence_col] = intersections[index_col].mod(NumBuckets) + 1
    else:
        # populate same value into all intersections
        intersections[sequence_col] = 1

    # logger.info("----- intersections (after adding sequence)---------")
    # logger.info("\n{}".format(intersections.to_csv(index=False)))

    return intersections


def get_sequence_with_partition_updated(
    intersections: pd.DataFrame,
    partition_by: list,
    sequence_col: str,
    num_intersections_in_one_slice: int,
):
    # Sort by partition_by to respect partitions
    # Group by the specified columns and calculate the size of each group
    group_sizes = intersections.groupby(partition_by).size().reset_index(name="size")
    # Sort groups by size in descending order
    sorted_groups = group_sizes.sort_values(by="size", ascending=False).reset_index(drop=True)
    # Calculate cumulative sum of sizes
    sorted_groups["cumulative_size"] = sorted_groups["size"].cumsum()
    # Calculate sequence numbers based on cumulative sizes
    sorted_groups["initial_sequence"] = (
        sorted_groups["cumulative_size"] // num_intersections_in_one_slice
    ).astype(int)
    # Reindex sequences to be consecutive from 1 to n
    sorted_groups[sequence_col] = sorted_groups["initial_sequence"].rank(method="dense").astype(int)
    # Merge the sequence information back to the original DataFrame
    result = intersections.merge(
        sorted_groups[partition_by + [sequence_col]],
        on=partition_by,
        how="left",
    )
    return result


def get_sequence_with_partition(
    intersections: pd.DataFrame,
    num_intersections_in_one_slice: int,
    partition_by: list,
    index_col: str,
    sequence_col: str,
    respect_partition_over_slice_size=False,
) -> pd.DataFrame:
    if respect_partition_over_slice_size:
        return get_sequence_with_partition_updated(
            intersections=intersections,
            partition_by=partition_by,
            sequence_col=sequence_col,
            num_intersections_in_one_slice=num_intersections_in_one_slice,
        )

    all_data = list()
    counter = 0
    prev_max_value_of_slice = 0
    # groupby location group and assign sequences
    for the_name, the_group in intersections.groupby(partition_by):
        the_group.reset_index(drop=True, inplace=True)

        TotalNumOfIntersections = len(the_group)

        if TotalNumOfIntersections < num_intersections_in_one_slice:
            NumBuckets = 1
        else:
            # determine the number of slice buckets by diving (whole number division) total num of intersections by NumIntersectionsInOneSlice
            # add 1 to ensure that there's atleast one bucket
            NumBuckets = (TotalNumOfIntersections // num_intersections_in_one_slice) + 1

        logger.debug(f"--- {partition_by} : {the_name}, NumBuckets : {NumBuckets}")

        # reset index to have continuous indexes from 0 to n
        the_group.reset_index(drop=True, inplace=True)

        # # reset index again to obtain index column
        the_group.reset_index(inplace=True)

        if NumBuckets > 1:
            # take modulus of row number by NumBuckets to assign each row to a different slice sequence
            the_group[sequence_col] = the_group[index_col].mod(NumBuckets) + 1
        else:
            # populate same value into all intersections
            the_group[sequence_col] = 1

        if counter > 0:
            the_group[sequence_col] = the_group[sequence_col] + prev_max_value_of_slice

        # store the existing iteration value for next iteration
        prev_max_value_of_slice = prev_max_value_of_slice + NumBuckets

        all_data.append(the_group)
        counter += 1

    intersections = pd.concat(all_data, ignore_index=True)

    return intersections


def assign_sequence(
    df: pd.DataFrame,
    grains: list,
    NumIntersectionsInOneSlice: int,
    index_col: str,
    sequence_col: str,
    slice_association_col: str,
    cols_required_in_output: list,
    partition_by: list,
) -> pd.DataFrame:

    logger.info(f"grains : {grains}")
    logger.info(f"partition_by : {partition_by}")
    logger.info(f"slice_association_col : {slice_association_col}")

    logger.info("shape : {}".format(df.shape))
    df.sort_values(grains, inplace=True)

    partitionby_cols_available = set(partition_by).issubset(set(list(df.columns)))
    if not partitionby_cols_available:
        logger.warning(f"One or more columns in {partition_by} not found in df")

    if partition_by and partitionby_cols_available:
        intersections = get_sequence_with_partition(
            intersections=df,
            num_intersections_in_one_slice=NumIntersectionsInOneSlice,
            partition_by=partition_by,
            index_col=index_col,
            sequence_col=sequence_col,
            respect_partition_over_slice_size=True,
        )
    else:
        intersections = get_sequence(
            intersections=df,
            num_intersections_in_one_slice=NumIntersectionsInOneSlice,
        )

    # type conversion to string to match dim definition on the tenant
    intersections[sequence_col] = intersections[sequence_col].astype("str")

    intersections[slice_association_col] = 1.0
    Output = intersections[cols_required_in_output]
    return Output


col_mapping = {
    "Slice Association Stat": float,
    "Slice Association TL": float,
    "Slice Association PL": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    StatGrains,
    NumIntersectionsInOneSlice,
    StatActuals,
    ForecastIterationSelection,
    TransitionGrains,
    PlanningGrains,
    StatPartitionBy,
    TransitionPartitionBy,
    PlanningPartitionBy,
    ForecastLevelData=None,
    ItemMasterData=None,
    RegionMasterData=None,
    AccountMasterData=None,
    ChannelMasterData=None,
    PnLMasterData=None,
    DemandDomainMasterData=None,
    LocationMasterData=None,
    ForecastIterationMasterData=None,
    AssortmentStat=None,
    AssortmentCustom=None,
    default_mapping={},
):
    try:
        StatOutputList = list()
        TransitionOutputList = list()
        PlanningOutputList = list()
        for the_iteration in StatActuals[o9Constants.FORECAST_ITERATION].unique():
            logger.warning(f"--- Processing iteration {the_iteration}")

            decorated_func = filter_for_iteration(iteration=the_iteration)(processIteration)

            (
                the_stat_output,
                the_transition_output,
                the_planning_output,
            ) = decorated_func(
                StatGrains=StatGrains,
                NumIntersectionsInOneSlice=NumIntersectionsInOneSlice,
                StatActuals=StatActuals,
                ForecastIterationSelection=ForecastIterationSelection,
                TransitionGrains=TransitionGrains,
                PlanningGrains=PlanningGrains,
                StatPartitionBy=StatPartitionBy,
                TransitionPartitionBy=TransitionPartitionBy,
                PlanningPartitionBy=PlanningPartitionBy,
                ForecastLevelData=ForecastLevelData,
                ItemMasterData=ItemMasterData,
                RegionMasterData=RegionMasterData,
                AccountMasterData=AccountMasterData,
                ChannelMasterData=ChannelMasterData,
                PnLMasterData=PnLMasterData,
                DemandDomainMasterData=DemandDomainMasterData,
                LocationMasterData=LocationMasterData,
                ForecastIterationMasterData=ForecastIterationMasterData,
                AssortmentStat=AssortmentStat,
                AssortmentCustom=AssortmentCustom,
                the_iteration=the_iteration,
                default_mapping=default_mapping,
            )

            StatOutputList.append(the_stat_output)
            TransitionOutputList.append(the_transition_output)
            PlanningOutputList.append(the_planning_output)

        StatOut = concat_to_dataframe(StatOutputList)
        TransitionOut = concat_to_dataframe(TransitionOutputList)
        PlanningOut = concat_to_dataframe(PlanningOutputList)
    except Exception as e:
        logger.exception(e)
        StatOut, TransitionOut, PlanningOut = None, None, None
    return StatOut, TransitionOut, PlanningOut


def add_parent_grains(
    df,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
):
    if ForecastLevelData is None:
        return df

    # add stat grains to history
    master_data_dict = {}
    master_data_dict["Item"] = ItemMasterData
    master_data_dict["Channel"] = ChannelMasterData
    master_data_dict["Demand Domain"] = DemandDomainMasterData
    master_data_dict["Region"] = RegionMasterData
    master_data_dict["Account"] = AccountMasterData
    master_data_dict["PnL"] = PnLMasterData
    master_data_dict["Location"] = LocationMasterData

    level_cols = [x for x in ForecastLevelData.columns if "Level" in x]
    for the_col in level_cols:

        # extract 'Item' from 'Item Level'
        the_dim = the_col.split(" Level")[0]

        logger.debug(f"the_dim : {the_dim}")

        # all dims exception location will be planning location
        if the_dim in ["Item", "Demand Domain"]:
            the_child_col = the_dim + ".[Transition " + the_dim + "]"
        else:
            the_child_col = the_dim + ".[Planning " + the_dim + "]"

        logger.debug(f"the_planning_col : {the_child_col}")

        # Item.[Stat Item]
        the_stat_col = the_dim + ".[Stat " + the_dim + "]"
        logger.debug(f"the_stat_col : {the_stat_col}")

        the_dim_data = master_data_dict[the_dim]

        # Eg. the_level = Item.[L2]
        the_level = (
            the_dim
            + ".["
            + add_dim_suffix(input=ForecastLevelData[the_col].iloc[0], dim=the_dim)
            + "]"
        )
        logger.debug(f"the_level : {the_level}")

        # copy values from L2 to Stat Item
        the_dim_data[the_stat_col] = the_dim_data[the_level]

        # select only relevant columns
        the_dim_data = the_dim_data[[the_child_col, the_stat_col]].drop_duplicates()

        # join with Actual
        df = df.merge(the_dim_data, on=the_child_col, how="inner")

    return df


def check_all_dimensions(df, grains, default_mapping):
    # checks if all 7 dim are present, if not, adds the dimension with member "All"
    # Renames input stream to Actual as well
    df_copy = df.copy()
    dims = {}
    for x in grains:
        dims[x] = x.strip().split(".")[0]
    try:
        for i in dims:
            if i not in df_copy.columns:
                if dims[i] in default_mapping:
                    df_copy[i] = default_mapping[dims[i]]
                else:
                    logger.warning(
                        f'Column {i} not found in default_mapping dictionary, adding the member "All"'
                    )
                    df_copy[i] = "All"
    except Exception as e:
        logger.exception(f"Error in check_all_dimensions\nError:-{e}")
        return df
    return df_copy


def processIteration(
    StatGrains,
    NumIntersectionsInOneSlice,
    StatActuals,
    ForecastIterationSelection,
    TransitionGrains,
    PlanningGrains,
    StatPartitionBy,
    TransitionPartitionBy,
    PlanningPartitionBy,
    ForecastLevelData,
    ItemMasterData,
    RegionMasterData,
    AccountMasterData,
    ChannelMasterData,
    PnLMasterData,
    DemandDomainMasterData,
    LocationMasterData,
    ForecastIterationMasterData,
    AssortmentStat,
    AssortmentCustom,
    the_iteration,
    default_mapping={},
):
    plugin_name = "DP041PopulateSliceAssociation"
    logger.info("Executing {} ...".format(plugin_name))

    version_col = "Version.[Version Name]"
    sequence_col = "Sequence.[Sequence]"
    slice_association_stat_col = "Slice Association Stat"
    slice_association_tl_col = "Slice Association TL"
    slice_association_pl_col = "Slice Association PL"
    index_col = "index"

    stat_forecast_level = get_list_of_grains_from_string(input=StatGrains)

    logger.info("Stat forecast_level : {}".format(stat_forecast_level))

    transition_forecast_level = get_list_of_grains_from_string(input=TransitionGrains)

    logger.info("Transition forecast_level : {}".format(transition_forecast_level))

    pl_forecast_level = get_list_of_grains_from_string(input=PlanningGrains)

    logger.info("Planning forecast_level : {}".format(pl_forecast_level))

    StatPartitionBy = get_list_of_grains_from_string(input=StatPartitionBy)

    # if no input supplied by user, override with defaults
    if TransitionPartitionBy.lower() in ["none", "na", ""]:
        tl_default_partitionby = "Location.[Stat Location],Account.[Stat Account],Channel.[Stat Channel],Region.[Stat Region],PnL.[Stat PnL],Demand Domain.[Stat Demand Domain],Item.[Stat Item]"
        logger.warning(f"Overriding TransitionPartitionBy to {tl_default_partitionby}")
        TransitionPartitionBy = tl_default_partitionby
    TransitionPartitionBy = get_list_of_grains_from_string(input=TransitionPartitionBy)

    # if no input supplied by user, override with defaults
    if PlanningPartitionBy.lower() in ["none", "na", ""]:
        pl_default_partitionby = "Location.[Planning Location],Account.[Planning Account],Channel.[Planning Channel],Region.[Planning Region],PnL.[Planning PnL],Demand Domain.[Transition Demand Domain],Item.[Transition Item]"
        logger.warning(f"Overriding PlanningPartitionBy to {pl_default_partitionby}")
        PlanningPartitionBy = pl_default_partitionby

    PlanningPartitionBy = get_list_of_grains_from_string(input=PlanningPartitionBy)

    logger.info(f"StatPartitionBy : {StatPartitionBy}")
    logger.info(f"TransitionPartitionBy : {TransitionPartitionBy}")
    logger.info(f"PlanningPartitionBy : {PlanningPartitionBy}")

    # select required columns
    stat_req_cols = [version_col] + stat_forecast_level + [sequence_col, slice_association_stat_col]

    transition_req_cols = (
        [version_col] + transition_forecast_level + [sequence_col, slice_association_tl_col]
    )

    pl_req_cols = [version_col] + pl_forecast_level + [sequence_col, slice_association_pl_col]

    StatOutput = pd.DataFrame(columns=stat_req_cols)
    TransitionOutput = pd.DataFrame(columns=transition_req_cols)
    PlanningOutput = pd.DataFrame(columns=pl_req_cols)
    try:
        input_stream = None
        if len(ForecastIterationMasterData) > 0:
            input_stream = ForecastIterationMasterData["Iteration Type Input Stream"].values[0]
        if input_stream is None:
            logger.warning(
                f"Invalid input stream 'None' for the forecast iteration {the_iteration}, returning empty outputs"
            )
            return StatOutput, TransitionOutput, PlanningOutput

        logger.info("NumIntersectionsInOneSlice : {}".format(NumIntersectionsInOneSlice))
        logger.info("Converting NumIntersectionsInOneSlice to integer ...")
        NumIntersectionsInOneSlice = int(NumIntersectionsInOneSlice)

        assert (
            NumIntersectionsInOneSlice > 0
        ), "NumIntersectionsInOneSlice should be positive integer ..."

        if stat_forecast_level and not StatActuals.empty:
            StatOutput = assign_sequence(
                df=StatActuals,
                grains=stat_forecast_level,
                NumIntersectionsInOneSlice=NumIntersectionsInOneSlice,
                index_col=index_col,
                sequence_col=sequence_col,
                slice_association_col=slice_association_stat_col,
                cols_required_in_output=stat_req_cols,
                partition_by=StatPartitionBy,
            )

        if input_stream == "Sell In Actual" or input_stream == "Actual":
            AssortmentFinal = AssortmentStat
        elif (AssortmentCustom is not None) and len(AssortmentCustom) > 0:
            history_dims = [col.split(".")[0] for col in AssortmentCustom]
            target_dims = [dim.split(".")[0] for dim in transition_forecast_level]
            missing_dims = list(set(target_dims) - set(history_dims))
            missing_grains = list(set(transition_forecast_level) - set(AssortmentCustom.columns))
            missing_dim_grains = []
            if len(missing_dims) > 0:
                for dim in missing_dims:
                    missing_dim_grains += [
                        col for col in transition_forecast_level if col.split(".")[0] == dim
                    ]
            if len(missing_dim_grains) > 0:
                AssortmentCustom = check_all_dimensions(
                    df=AssortmentCustom,
                    grains=missing_dim_grains,
                    default_mapping=default_mapping,
                )
            missing_grains = list(set(missing_grains) - set(missing_dim_grains))

            # add stat grains to history
            master_data_dict = {}
            master_data_dict["Item"] = ItemMasterData
            master_data_dict["Channel"] = ChannelMasterData
            master_data_dict["Demand Domain"] = DemandDomainMasterData
            master_data_dict["Region"] = RegionMasterData
            master_data_dict["Account"] = AccountMasterData
            master_data_dict["PnL"] = PnLMasterData
            master_data_dict["Location"] = LocationMasterData

            # if target grain is missing, but there is another level present in input for the same dimension, we go through the master data and find the target grain values
            if len(missing_grains) > 0:
                for grain in missing_grains:
                    dim_of_missing_grain = grain.split(".")[0]
                    master_df = master_data_dict[dim_of_missing_grain]
                    existing_grain = [
                        col
                        for col in AssortmentCustom.columns
                        if col.split(".")[0] == dim_of_missing_grain
                    ][0]
                    AssortmentCustom = AssortmentCustom.merge(
                        master_df[[existing_grain, grain]],
                        on=existing_grain,
                        how="inner",
                    )

            AssortmentFinal = AssortmentCustom
        else:
            if the_iteration == "FI-PL":
                logger.warning(
                    "Assortment measure not populated for the forecast iteration FI-PL, returning empty dataframes for Transition and Planning Outputs"
                )
                return StatOutput, TransitionOutput, PlanningOutput

        if transition_forecast_level:
            if the_iteration == "FI-PL":
                transition_intersections = AssortmentFinal[
                    [version_col] + transition_forecast_level
                ].drop_duplicates()
            else:
                transition_intersections = ForecastIterationSelection[
                    [version_col] + transition_forecast_level
                ].drop_duplicates()

            transition_intersections = add_parent_grains(
                df=transition_intersections,
                ForecastLevelData=ForecastLevelData,
                ItemMasterData=ItemMasterData,
                RegionMasterData=RegionMasterData,
                AccountMasterData=AccountMasterData,
                ChannelMasterData=ChannelMasterData,
                PnLMasterData=PnLMasterData,
                DemandDomainMasterData=DemandDomainMasterData,
                LocationMasterData=LocationMasterData,
            )

            TransitionOutput = assign_sequence(
                df=transition_intersections,
                grains=transition_forecast_level,
                NumIntersectionsInOneSlice=NumIntersectionsInOneSlice,
                index_col=index_col,
                sequence_col=sequence_col,
                slice_association_col=slice_association_tl_col,
                cols_required_in_output=transition_req_cols,
                partition_by=TransitionPartitionBy,
            )

        if pl_forecast_level:
            req_dim_cols = list(set(pl_forecast_level) | set(transition_forecast_level))
            if the_iteration == "FI-PL":
                planning_intersections = AssortmentFinal[
                    [version_col] + req_dim_cols
                ].drop_duplicates()
            else:
                planning_intersections = ForecastIterationSelection[
                    [version_col] + req_dim_cols
                ].drop_duplicates()

            PlanningOutput = assign_sequence(
                df=planning_intersections,
                grains=pl_forecast_level,
                NumIntersectionsInOneSlice=NumIntersectionsInOneSlice,
                index_col=index_col,
                sequence_col=sequence_col,
                slice_association_col=slice_association_pl_col,
                cols_required_in_output=pl_req_cols,
                partition_by=PlanningPartitionBy,
            )

        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(e)
        StatOutput = pd.DataFrame(columns=stat_req_cols)
        TransitionOutput = pd.DataFrame(columns=transition_req_cols)
        PlanningOutput = pd.DataFrame(columns=pl_req_cols)

    return StatOutput, TransitionOutput, PlanningOutput
