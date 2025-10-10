import logging

import pyspark
from o9Reference.common_utils.decorators import (
    map_output_columns_to_dtypes,  # type: ignore
)
from o9Reference.common_utils.function_logger import (
    log_inputs_and_outputs,  # type: ignore
)
from o9Reference.spark_utils.common_utils import get_clean_string  # type: ignore
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql.functions import sum

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
    """Function to aggregate the input data to the output level, given that the output grain is at a higher level than the Input
    Aggregate + Copy template
    Args:
        SourceMeasures (str): 1 Source Measure
        SourceGrain (str): Source grains seperated by commas
        TargetMeasures (str): 1 target measure
        TargetGrain (str): Target grains seperated by commas
        Aggregation (str): default - SUM
        Input (pyspark.sql.dataframe.DataFrame): Input pyspark sql dataframe

    Returns:
        pyspark.sql.dataframe.DataFrame: Output dataframe
    """
    plugin_name = "PopulateBaseFcstQty_Pyspark"
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

        logger.debug(f"target_measures : {target_measures}, {len(target_measures)}")
        # check if target grains are present in the Input data so aggregation can be done to the output grain
        if all(col not in Input.columns for col in target_grains):
            raise ValueError("All the Target Grains are not present in the Input Data... ")

        # check the length of the source and target measures
        check_length_source = (
            len(source_measures) > 1,
            f"Aggregation is done for only one target measure.. {source_measures}",
        )
        assert check_length_source

        check_length_target = (
            len(target_measures) > 1,
            f"Aggregation is done for only one target measure.. {target_measures}",
        )
        assert check_length_target

        # Aggregate the Input Data to target grain level for the target measure
        if Aggregation.upper() == "SUM":
            Output = Input.groupBy(*target_grains).agg(
                sum(source_measures[0]).alias(target_measures[0])
            )

        Output = col_namer.convert_to_o9_cols(df=Output)
        return Output
    except Exception as e:
        logger.exception(e)
        raise e
