import logging

from o9Reference.common_utils.decorators import map_output_columns_to_dtypes
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer

from helpers.DP070FlexibleLevelLocationSplit import df_keys
from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()

# TODO : Fill this with output column list
col_mapping = {}


@log_inputs_and_outputs
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(Input, InputMeasures, OutputMeasures):
    # TODO : Change plugin name here
    plugin_name = "PysparkCopyTemplate"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        # get clean list of input and output measures
        input_measures = get_list_of_grains_from_string(input=InputMeasures)
        output_measures = get_list_of_grains_from_string(input=OutputMeasures)

        logger.debug(f"input_measures : {input_measures}")
        logger.debug(f"output_measures : {output_measures}")

        # assert list provided is of the same length
        assert len(input_measures) == len(
            output_measures
        ), "Number of entries in InputMeasures should match OutputMeasures ..."

        # Assert that all columns_to_check are in actual_columns
        assert all(
            col in Input.columns for col in input_measures
        ), "One or more columns specified in InputMeasures not included in Input query ..."

        # form a dictionary
        output_to_input_mapping = dict(zip(output_measures, input_measures))

        for the_output, the_input in output_to_input_mapping.items():
            logger.debug(f"Copying {the_input} to {the_output} ...")
            # Copy values from one column to another
            Input = Input.withColumn(the_output, Input[the_input])

        # drop the input measures
        Output = Input.drop(*input_measures)

        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {} for slice : {}".format(e, df_keys))
    return Output
