import logging

from o9Reference.common_utils.decorators import map_output_columns_to_dtypes
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from pyspark.sql.functions import lit

from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

# TODO : Fill this with output column list
col_mapping = {}


@log_inputs_and_outputs
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(Input, MeasuresToNullOut):
    plugin_name = "PysparkNullOutTemplate"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        # get clean list
        measures_to_null_out = get_list_of_grains_from_string(input=MeasuresToNullOut)
        for the_measure in measures_to_null_out:
            # Get the data type of the input column
            the_input_dtype = Input.schema[the_measure].dataType
            logger.debug(f"the_input_dtype : {the_input_dtype}")
            logger.debug(f"Nulling out {the_measure}")
            # Null out the column
            Input = Input.withColumn(the_measure, lit(None).cast(the_input_dtype))

    except Exception as e:
        logger.exception(e)
    return Input
