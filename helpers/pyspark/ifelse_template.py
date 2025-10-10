import logging

from o9Reference.common_utils.decorators import map_output_columns_to_dtypes
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer, get_clean_string
from pyspark.sql.functions import col as fcol
from pyspark.sql.functions import when

from helpers.utils import get_list_of_grains_from_string

col_namer = ColumnNamer()

logger = logging.getLogger("o9_logger")

# TODO : Fill this with output column list
col_mapping = {}


@log_inputs_and_outputs
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(Input, OutputMeasure, If, Is, Then, Else):
    plugin_name = "PysparkIfElseTemplate"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        assert Is.lower() in [
            "null",
            "notnull",
        ], f"Invalid entry {Is} Allowed values in 'Is' are 'NULL' and 'NOTNULL' ..."

        # get clean list
        input_measures = get_list_of_grains_from_string(input=",".join([If, Then, Else]))

        # drop duplicates
        input_measures = list(set(input_measures))

        logger.debug(f"input_measures : {input_measures}")

        # select all columns excluding input measures
        dim_cols_required_in_output = [x for x in Input.columns if x not in input_measures]
        logger.debug(f"dim_cols_required_in_output : {dim_cols_required_in_output}")

        logger.debug(f"Input.columns : {Input.columns}")
        logger.debug(f"input_measures : {input_measures}")

        Input = col_namer.convert_to_pyspark_cols(df=Input)

        # Assert that all columns_to_check are in actual_columns
        assert all(
            col in Input.columns for col in input_measures
        ), "One or more columns specified in InputMeasures not included in Input query ..."

        if Is == "null":
            Input = Input.withColumn(
                OutputMeasure,
                when(fcol(If).isNull(), fcol(Then)).otherwise(fcol(Else)),
            )
        else:
            Input = Input.withColumn(
                OutputMeasure,
                when(fcol(If).isNotNull(), fcol(Then)).otherwise(fcol(Else)),
            )

        cols_required_in_output = [get_clean_string(x) for x in dim_cols_required_in_output] + [
            OutputMeasure
        ]
        # drop the input measures
        Output = Input.select(*cols_required_in_output)

        Output = col_namer.convert_to_o9_cols(df=Output)

        # Your code ends here
        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.exception("Exception {}".format(e))
    return Output
