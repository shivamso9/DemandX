import logging
from functools import reduce

from dateutil.parser import parse
from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, to_timestamp, trim, udf
from pyspark.sql.types import DoubleType, TimestampType

from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


def parse_and_format_to_standard(date_str):
    try:
        dt = parse(date_str)
        return dt
    except Exception:
        return None


parse_udf = udf(parse_and_format_to_standard, TimestampType())


def parse_mixed_timestamp(df, column, output_column=None):
    if output_column is None:
        output_column = column
    return df.withColumn(output_column, parse_udf(col(column)))


def find_new_and_modified_records(
    Output, OutputDataframe, DateColumn, DPIncrementalMeasure, PromoGrains=None
):
    try:
        logger.info("Executing find_new_and_modified_records")
        output_measures = get_list_of_grains_from_string(input=DateColumn)
        if OutputDataframe.count() == 0:
            logger.info("OutputDataframe is empty returning Output")
            return Output
        logger.info("Fetching new/modified intersections")
        filtered_Df = OutputDataframe.filter(OutputDataframe[DPIncrementalMeasure] == 1)
        OutputDataframe = OutputDataframe.withColumn(DPIncrementalMeasure, lit(None))
        OutputDataframe = col_namer.convert_to_pyspark_cols(OutputDataframe)
        required_column = col_namer.convert_to_pyspark_cols(Output).columns + [DPIncrementalMeasure]
        OutputDataframe = OutputDataframe.select(required_column)
        OutputDataframe = col_namer.convert_to_o9_cols(OutputDataframe)
        date_columns = [col for col in output_measures if col in Output.columns]
        start_date, end_date = get_list_of_grains_from_string(DateColumn)

        Output = Output.withColumn(start_date, trim(col(start_date).cast("string")))
        Output = Output.withColumn(end_date, trim(col(end_date).cast("string")))

        OutputDataframe = OutputDataframe.withColumn(
            start_date, trim(col(start_date).cast("string"))
        )
        OutputDataframe = OutputDataframe.withColumn(end_date, trim(col(end_date).cast("string")))

        Output = parse_mixed_timestamp(Output, start_date)
        Output = parse_mixed_timestamp(Output, end_date)
        OutputDataframe = parse_mixed_timestamp(OutputDataframe, start_date)
        OutputDataframe = parse_mixed_timestamp(OutputDataframe, end_date)

        new_entries = Output.join(OutputDataframe, on=Output.columns, how="left_anti")

        new_entries = new_entries.withColumn(DPIncrementalMeasure, lit(1))
        join_keys = [
            col for col in Output.columns if col not in output_measures + [DPIncrementalMeasure]
        ]
        logger.info(new_entries.count())

        modified_condition = reduce(
            lambda x, y: x | y, [col(f"df1.{c}") != col(f"df2.{c}") for c in date_columns]
        )
        Output = Output.withColumn(DPIncrementalMeasure, lit(None))
        modified_records = (
            Output.alias("df1")
            .join(OutputDataframe.alias("df2"), on=join_keys, how="inner")
            .filter(modified_condition)
            .select("df1.*")
        )
        logger.info(modified_records.count())
        modified_records = modified_records.withColumn(DPIncrementalMeasure, lit(1))
        result = OutputDataframe.join(modified_records, on=join_keys, how="left_anti")
        Output = result.union(new_entries).union(modified_records).dropDuplicates(join_keys)
        Output = Output.withColumn(
            DPIncrementalMeasure, col(DPIncrementalMeasure).cast(DoubleType())
        )
        Output = Output.withColumn(
            start_date, to_timestamp(start_date, "M/d/yyyy h:mm:ss a")
        ).withColumn(end_date, to_timestamp(end_date, "M/d/yyyy h:mm:ss a"))
        Output = Output.filter(col(DPIncrementalMeasure) == 1.0)
        filtered_Df = filtered_Df.withColumn(DPIncrementalMeasure, lit(None).cast(DoubleType()))
        Output = Output.unionByName(filtered_Df).dropDuplicates()
        exclude = [start_date, end_date, DPIncrementalMeasure]
        if PromoGrains is not None:
            PromoGrains = get_list_of_grains_from_string(PromoGrains)
            exclude.extend(PromoGrains)
        key_columns = [col for col in Output.columns if col not in exclude]
        Output = Output.orderBy(F.desc(DPIncrementalMeasure)).dropDuplicates(subset=key_columns)
        logger.info("Successfully executed ")
    except Exception as e:
        logger.exception(f"Exception for slice : {e}")
    return Output
