import logging

import pyspark
from o9Reference.common_utils.decorators import (
    map_output_columns_to_dtypes,  # type: ignore
)
from o9Reference.common_utils.function_logger import (
    log_inputs_and_outputs,  # type: ignore
)
from o9Reference.spark_utils.common_utils import ColumnNamer  # type: ignore
from o9Reference.spark_utils.common_utils import (  # type: ignore
    get_clean_string,
    is_dimension,
)

from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()

# TODO : Fill this with output column list
col_mapping = {}


@log_inputs_and_outputs
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    InputDataDict: dict,  # InputMultiply1, InputMultiply2
    InputMeasures: str,  # list of measures from that needs to be multiplied
    OutputMeasure: str,
    # spark_session,
) -> pyspark.sql.dataframe.DataFrame:
    """Main function to multiply measures in diff measure groups and aggregate it to a common level

    Args:
        InputDataDict (dict): Dictionary which consists of Input dataframes that are to be multiplied.
        InputMeasures (str): List of input measures to be multiplied corresponding to Input dictionary, i.e.,
            InputDataDict.value[0] -> InputMeasures[0]
            InputDataDict.value[1] -> InputMeasures[1], input data and the corresponding measure that needs to be multiplied

    Returns:
        Output: Pyspark DataFrame
    """
    plugin_name = "PopulateBaseFcstQtyInt_Pyspark"
    logger.info("Executing {} ...".format(plugin_name))
    # spark = spark_session
    try:
        # OutputData = col_namer.convert_to_pyspark_cols(OutputData)

        source_measures = get_list_of_grains_from_string(input=InputMeasures)
        # source_measures = [get_clean_string(x) for x in source_measures]
        logger.debug(f"source_measures : {source_measures}")

        target_measure = get_list_of_grains_from_string(input=OutputMeasure)
        # target_measure = [get_clean_string(x) for x in target_measure]
        logger.debug(f"target_measures : {target_measure}")

        input_measure_dict = {}
        # output_data = spark.createDataFrame([], StructType([]))
        logger.debug("------------")
        for idx, (key, input_data) in enumerate(InputDataDict.items()):

            # Check the grains of the input data set to be multiplied
            grain_list = [get_clean_string(col) for col in input_data.columns if is_dimension(col)]
            input_measure_dict[idx] = grain_list
            logger.debug(f"input_measure_dict : {input_measure_dict}")

            input_data = col_namer.convert_to_pyspark_cols(input_data)

            logger.debug(f"input data {idx} rows count : {input_data.count()}")
            logger.debug(f"input data {idx} colums : {input_data.columns}")
            logger.debug(f"input data {idx} dtypes : {input_data.dtypes}")

            # if output_data.rdd.isEmpty():
            if idx == 0:
                output_data = input_data
                logger.debug(f"output data {idx} rows count : {output_data.count()}")
                logger.debug(f"output data {idx} colums : {output_data.columns}")

            # check if the columns are present in the measure groups
            # source_measure[0] --> input_data[0], i.e., first set of measure in the list should be present in the first measure group and so on..
            if source_measures[idx] not in input_data.columns:
                raise ValueError(
                    f"missing_columns : {source_measures[idx]} not present in the {input_data}, {idx + 1} Measure Group, and check the order of inputs in the plugin settings..."
                )

            # Check if the grains in all the measures are of the same level
            if idx > 0:
                check_grains = (
                    len(set(input_measure_dict[idx])) - len(set(input_measure_dict[idx - 1])) != 0,
                    "grains are not of the same level",
                )
                assert check_grains

                # Join inputs with each other if they are at the same level
                common_cols = set(input_measure_dict[idx]).intersection(input_measure_dict[idx - 1])
                logger.debug(f"common_cols are : {list(common_cols)}")

                join_col = list(common_cols)
                output_data = output_data.join(input_data, on=join_col, how="inner")
                logger.debug(f"join data rows count : {output_data.count()}")
                logger.debug(f"join data columns count : {output_data.columns}")
                logger.debug(f"join data dtypes : {output_data.dtypes}")

        # multiply the input measures
        # check if the columns to be multiplied are present in the final joined data
        if all(col not in output_data.columns for col in source_measures):
            raise ValueError(
                "Measures are not present in the input data.. check the InputMeasures and the Input datasets.."
            )

        #  multiply the measures and return output
        check_len_measures = (
            len(source_measures) > len(target_measure),
            "Length of output measures must be less than input measures..",
        )
        assert check_len_measures

        for i in range(1, len(source_measures)):
            output_data = output_data.withColumn(
                target_measure[i - 1],
                output_data[source_measures[i - 1]] * output_data[source_measures[i]],
            )

        # drop the source measures and keep the target measures
        Output = output_data.drop(*source_measures)
        Output = col_namer.convert_to_o9_cols(Output)

        logger.debug(f"Output rows count : {Output.count()}")
        logger.debug(f"Output columns count : {Output.columns}")
        logger.debug(f"Output dtypes : {Output.dtypes}")
        return Output
    except Exception as e:
        raise e
