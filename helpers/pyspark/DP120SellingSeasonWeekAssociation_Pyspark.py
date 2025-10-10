import logging

import pyspark.sql.functions as F
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType

logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


# Define main function
@log_inputs_and_outputs
def main(
    AssortmentDates,
    Time,
):

    plugin_name = "DP120SellingSeasonWeekAssociation_Pyspark"
    logger.info(f"Executing {plugin_name} ...")

    try:
        Time = col_namer.convert_to_pyspark_cols(Time)
        AssortmentDates = col_namer.convert_to_pyspark_cols(AssortmentDates)

        # Validate input DataFrames
        if AssortmentDates.isEmpty():
            raise ValueError(
                "AssortmentDates dataframe cannot be empty! Check logs/ inputs for error!"
            )
        if Time.isEmpty():
            raise ValueError("Time master data cannot be empty! Check logs/ inputs for error!")

        # Create DF1 - Extract distinct Intro and Disco Date combinations
        DF1 = AssortmentDates.select("Intro Date", "Disco Date").distinct()
        logger.info(f"DF1 count: {DF1.count()}")

        # Create DF2 - Join Time with DF1 to get PartialWeekKey
        DF2 = (
            Time.alias("t")
            .join(
                DF1.alias("df1"),
                (F.col("t.Time_DayKey") >= F.col("df1.Intro Date"))
                & (F.col("t.Time_DayKey") <= F.col("df1.Disco Date")),
                "left",
            )
            .select("t.Time_PartialWeek", "df1.Intro Date", "df1.Disco Date")
        )
        DF2 = DF2.dropDuplicates()

        logger.info(f"DF2 count: {DF2.count()}")

        # Create DF3 - Inner join AssortmentDates with DF2
        DF3 = AssortmentDates.alias("ad").join(
            DF2.alias("df2"),
            (F.col("ad.Intro Date") == F.col("df2.Intro Date"))
            & (F.col("ad.Disco Date") == F.col("df2.Disco Date")),
            "inner",
        )

        logger.info(f"DF3 count: {DF3.count()}")

        DF4 = DF3.drop("Intro Date", "Disco Date", "Assortment Foundation").withColumn(
            "Selling Season to Week Association", lit(1).cast(DoubleType())
        )

        logger.info(f"DF4 count: {DF4.count()}")

        # Assuming a function `truncate_and_append` exists to handle truncation and appending
        # truncate_and_append('025_Selling_Season_Week_Association', DF4)

        # Convert output DataFrame to O9 column format
        Output = col_namer.convert_to_o9_cols(df=DF4)

        logger.info("Output created... Returning it to the caller function")
    except Exception as e:
        logger.exception(e)
        Output = None

    return Output
