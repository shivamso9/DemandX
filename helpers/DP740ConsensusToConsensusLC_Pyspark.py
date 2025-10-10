import logging

from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer

logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


# Define main function
@log_inputs_and_outputs
def main(
    ConsensusFcst,
    ConsensusFcstLC,
    spark,
):
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    plugin_name = "DP740ConsensusToConsensusLC_Pyspark"
    logger.info(f"Executing {plugin_name} ...")

    consFcst_col = "Consensus Fcst"
    consFcstLC_col = "Consensus Fcst LC"

    try:
        ConsensusFcst = col_namer.convert_to_pyspark_cols(ConsensusFcst)
        ConsensusFcstLC = col_namer.convert_to_pyspark_cols(ConsensusFcstLC)

        # Validate input DataFrames
        if ConsensusFcst.isEmpty():
            raise ValueError(
                "ConsensusFcst dataframe cannot be empty! Check logs/ inputs for error!"
            )

        if ConsensusFcstLC.count() > 0:
            ConsensusFcstLC = ConsensusFcstLC.filter("1 = 0")

        ConsensusFcstLC = ConsensusFcst.withColumnRenamed(consFcst_col, consFcstLC_col)

        Output = ConsensusFcstLC.select("*")

        logger.info(f"ConsensusFcstLC columns: {ConsensusFcstLC.columns}")
        logger.info(f"ConsensusFcstLC count: {ConsensusFcstLC.count()}")

        # Convert output DataFrame to O9 column format
        Output = col_namer.convert_to_o9_cols(Output)

        logger.info(f"Successfully executed {plugin_name} ...")
    except Exception as e:
        logger.exception(e)

    return Output
