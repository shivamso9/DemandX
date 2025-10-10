import logging

import pyspark
from o9Reference.common_utils.decorators import (
    map_output_columns_to_dtypes,  # type: ignore
)
from o9Reference.common_utils.function_logger import (
    log_inputs_and_outputs,  # type: ignore
)
from o9Reference.spark_utils.common_utils import ColumnNamer  # type: ignore
from o9Reference.spark_utils.common_utils import get_clean_string  # type: ignore
from pyspark.sql.functions import col, lit, sum, when
from pyspark.sql.types import DoubleType

from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()

# TODO : Fill this with output column list
col_mapping = {}


@log_inputs_and_outputs
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    SourceMeasures: str,
    SourceGrain: str,
    TargetMeasures: str,
    TargetGrain: str,
    Aggregation: str,
    Input: pyspark.sql.dataframe.DataFrame,
) -> pyspark.sql.dataframe.DataFrame:
    """Function to aggregate at the output grain and create association measure

    Args:
        SourceMeasures (str): one source measure for which association needs to be calculated
        SourceGrain (str): source grains in the Input data separated by commas
        TargetMeasures (str): one target measure name
        TargetGrain (str): target grains in the Output data
        Aggregation (str): default : SUM is supported
        Input (pyspark.sql.dataframe.DataFrame): Input spark dataframe

    Returns:
        pyspark.sql.dataframe.DataFrame: Output dataframe consisting of the association measure
    """
    plugin_name = "DP804PopulatePublishedFcstAssociation_Pyspark"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        Input = col_namer.convert_to_pyspark_cols(Input)

        source_grains = get_list_of_grains_from_string(input=SourceGrain)
        source_grains = [get_clean_string(x) for x in source_grains]
        logger.debug(f"source_grains : {source_grains}")

        target_grains = get_list_of_grains_from_string(input=TargetGrain)
        target_grains = [get_clean_string(x) for x in target_grains]
        logger.debug(f"target_grains : {target_grains}")

        source_measures = get_list_of_grains_from_string(input=SourceMeasures)
        target_measures = get_list_of_grains_from_string(input=TargetMeasures)

        if source_measures[0] not in Input.columns:
            raise ValueError(
                "source measure is not present in the input dataset.. check the Input dataset and rerun.."
            )

        # check the length of the target measures, it should be 1 as aggregation is done for a col
        check_len_source = (
            len(source_measures) > 1,
            "Only 1 target measure to be present for the aggregation to work! Check script input params!",
        )
        assert check_len_source

        check_len_target = (
            len(target_measures) > 1,
            "Only 1 target measure to be present for the aggregation to work! Check script input params!",
        )
        assert check_len_target

        # check if target grains are present in the Input data as it is a higher level
        if all(col not in Input.columns for col in target_grains):
            raise ValueError("Target Grains are not present in the Input dataset..")

        # Aggregate at the grains level
        grouped_cols = target_grains
        logger.debug(f"The group by cols are : {grouped_cols}")
        if Aggregation.upper() == "SUM":
            Output = Input.groupby(*grouped_cols).agg(
                sum(source_measures[0]).alias(source_measures[0])
            )
        else:
            raise ValueError(
                f"Aggregation value : {Aggregation} is not supported and should be in all UPPERCASE!"
            )

        # Check for the col which compares > 0 and puts 1 else
        logger.debug(f"SourceMeasures: {SourceMeasures}, source_measures : {source_measures}")
        logger.debug(f"target_measures : {target_measures}")
        Output = Output.withColumn(
            target_measures[0],
            when(col(source_measures[0]) > 0, 1).otherwise(lit(None)),
        )

        Output = Output.drop(*source_measures)
        logger.debug(f"Output columns : {Output.columns}")
        logger.debug(f"Output row count : {Output.count()}")

        # type cast the target measure
        Output = Output.withColumn(target_measures[0], col(target_measures[0]).cast(DoubleType()))
        Output = col_namer.convert_to_o9_cols(df=Output)

    except Exception as e:
        logger.exception(e)
        Output = None
    return Output
