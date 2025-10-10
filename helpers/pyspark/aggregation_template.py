import logging

import pyspark
from o9Reference.common_utils.function_logger import (
    log_inputs_and_outputs,  # type: ignore
)
from o9Reference.spark_utils.common_utils import ColumnNamer  # type: ignore
from o9Reference.spark_utils.common_utils import get_clean_string  # type: ignore
from pyspark.sql.functions import col, collect_set

from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()


@log_inputs_and_outputs
def main(
    SourceMeasures: str,
    SourceGrain: str,
    TargetMeasures: str,
    TargetGrain: str,
    Aggregation: str,
    Input: pyspark.sql.dataframe.DataFrame,
    MasterDataDict: dict,
) -> pyspark.sql.dataframe.DataFrame:
    """Get aggregated outputs based on these aggregation logic - SUM, MAX, MIN, AVG, AVGNONNULL & DISTINCTSUM.

    Args:
        SourceMeasures (str): Comma separated source measures
        SourceGrain (str): Source Grains of the Input
        TargetMeasures (str): Comma separated target measures
        TargetGrain (str): Required Grains of the Output
        Aggregation (str): Aggregation logic to use
        Input (pyspark.sql.dataframe.DataFrame): Input Dataset
        MasterDataDict (dict): Master data dictionary

    Returns:
        pyspark.sql.dataframe.DataFrame: Aggregated Output dataset
    """
    plugin_name = "PysparkAggregationTemplate"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        Input = col_namer.convert_to_pyspark_cols(Input)

        source_grains = get_list_of_grains_from_string(input=SourceGrain)
        source_grains = [get_clean_string(x) for x in source_grains]
        logger.debug(f"source_grains : {source_grains}")

        target_grains = get_list_of_grains_from_string(input=TargetGrain)
        target_grains = [get_clean_string(x) for x in target_grains]
        logger.debug(f"target_grains : {target_grains}")

        # join input with master data
        for the_key, the_master_data in MasterDataDict.items():
            the_master_data = col_namer.convert_to_pyspark_cols(df=the_master_data)

            # identify common column if any
            the_common_columns = list(set(Input.columns).intersection(the_master_data.columns))
            logger.debug(f"the_key : {the_key}, common_columns : {the_common_columns}")
            if the_common_columns:
                the_join_col = the_common_columns[0]
                logger.debug(f"the_join_col : {the_join_col}")

                Input = Input.join(the_master_data, on=the_join_col, how="inner")

        source_measures = get_list_of_grains_from_string(input=SourceMeasures)
        target_measures = get_list_of_grains_from_string(input=TargetMeasures)

        input_to_output_measure_dict = dict(zip(source_measures, target_measures))

        logger.debug(f"input_to_output_measure_dict : {input_to_output_measure_dict}")

        # check if all target grains are present in Input after join
        missing_columns = [x for x in target_grains if x not in Input.columns]
        logger.debug(f"missing_columns : {missing_columns}")

        if missing_columns:
            raise ValueError(
                f"missing columns : {missing_columns}, please ensure 'Master' dataframes are supplied for all of these ..."
            )

        if Aggregation.upper() == "SUM":
            Output = Input.groupBy(*target_grains).sum()
            rename_mapping = dict(zip([f"sum({x})" for x in source_measures], target_measures))
        elif Aggregation.upper() == "MAX":
            Output = Input.groupBy(*target_grains).max()
            rename_mapping = dict(zip([f"max({x})" for x in source_measures], target_measures))
        elif Aggregation.upper() == "MIN":
            Output = Input.groupBy(*target_grains).min()
            rename_mapping = dict(zip([f"min({x})" for x in source_measures], target_measures))
        elif Aggregation.upper() == "AVG":
            Output = Input.groupBy(*target_grains).avg()
            rename_mapping = dict(zip([f"avg({x})" for x in source_measures], target_measures))
        elif Aggregation.upper() == "AVGNONNULL":
            # Drop rows from the source measures that have null values
            logger.debug(Input.count())
            Input = Input.na.drop(subset=source_measures)
            logger.debug(Input.count())

            # Take the avg of the measures
            Output = Input.groupBy(*target_grains).avg()
            rename_mapping = dict(zip([f"avg({x})" for x in source_measures], target_measures))
        elif Aggregation.upper() == "DISTINCTSUM":

            # Create the aggregation expression for the source measures
            GroupedInput = Input.groupby(*target_grains)
            logger.debug(type(GroupedInput))

            # Function to apply collect_set on a grouped input
            def apply_collect_set(
                df: pyspark.sql.group.GroupedData, col: str
            ) -> pyspark.sql.dataframe.DataFrame:
                """Apply collect_set to the grouped input data.

                Args:
                    df (pyspark.sql.group.GroupedData): pyspark sql dataframe grouped data
                    col (str): column value to apply collect_set

                Returns:
                    Aggregated pyspark dataframe
                """
                return df.agg(collect_set(col).alias(col))

            if len(source_measures) < 1:
                raise ValueError("source_measures should have at least 1 value!")

            Output = apply_collect_set(GroupedInput, col=source_measures[0])

            for idx, col_val in enumerate(source_measures):
                logger.debug(idx)
                if idx > 0:
                    Output_new = apply_collect_set(GroupedInput, col=col_val)

                    # join with the above
                    join_cols = list(set(Output.columns).intersection(Output_new.columns))

                    logger.debug(join_cols)
                    Output = Output.join(Output_new, on=join_cols, how="inner")

                logger.debug(Output.dtypes)

            for col_val in source_measures:
                Output = Output.withColumn(
                    col_val,
                    col(col_val).cast("string"),
                )

            logger.debug(Output.dtypes)
            rename_mapping = dict(
                zip(
                    source_measures,
                    target_measures,
                )
            )
        else:
            raise ValueError(f"Aggregation : {Aggregation} not supported ...")

        logger.debug(f"rename_mapping : {rename_mapping}")

        # Rename columns
        for old_name, new_name in rename_mapping.items():
            Output = Output.withColumnRenamed(old_name, new_name)

        Output = col_namer.convert_to_o9_cols(df=Output)

    except Exception as e:
        logger.exception(e)
        Output = None
    return Output
