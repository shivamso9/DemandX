import logging

import pyspark
from o9Reference.common_utils.decorators import (
    map_output_columns_to_dtypes,  # type: ignore
)
from o9Reference.common_utils.function_logger import (
    log_inputs_and_outputs,  # type: ignore
)
from o9Reference.spark_utils.common_utils import (  # type: ignore
    ColumnNamer,
    get_clean_string,
)
from pyspark import StorageLevel

from helpers.utils import get_list_of_grains_from_string

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()

# TODO : Fill this with output column list
col_mapping = {}


@log_inputs_and_outputs
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    Actuals: pyspark.sql.dataframe.DataFrame,
    Time: pyspark.sql.dataframe.DataFrame,
    actual_grains: str,
    time_grains: str,
    time_join_key: str,
    required_time_level: str,
) -> pyspark.sql.dataframe.DataFrame:
    """Function to aggregate the actuals from day to partial week level. Aggregation (sum) is done on the measures specified in the actual_grains.

    Args:
        Actuals (pyspark.sql.dataframe.DataFrame): Actuals pyspark dataframe
        Time (pyspark.sql.dataframe.DataFrame): Time Dimension dataframe
        actual_grains (str): Measures to be used for actuals data
        time_grains (str): Measures to be used for Time master data
        time_join_key (str): Key to join the Actuals and the Time Dimension data
        required_time_level (str): Lowest level grain to be aggregated

    Returns:
        pyspark.sql.dataframe.DataFrame: Output dataframe to be uploaded to the delta lake
    """
    plugin_name = "DP110HistoryAggregation"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        # Check if the Inputs consists of data
        if Actuals.count() == 0:
            raise ValueError("Actuals dataframe cannot be empty! Check logs/ inputs for error!")
        if Time.count() == 0:
            raise ValueError("Time master data cannot be empty! Check logs/ inputs for error!")
        if len(actual_grains.strip()) == 0 or len(time_grains.strip()) == 0:
            raise ValueError("Input script parameters doesn't contain any value!")

        time_dim_cols = get_list_of_grains_from_string(time_grains)
        time_dim_cols = [get_clean_string(x) for x in time_dim_cols]

        actual_grains = get_list_of_grains_from_string(actual_grains)
        actual_grains_cols = [get_clean_string(x) for x in actual_grains]

        time_join_key = get_clean_string(time_join_key)
        required_time_level = get_clean_string(required_time_level)

        logger.debug(f"Time Dimension cols are : {time_dim_cols}")
        logger.debug(f"Actuals grains cols are : {actual_grains_cols} ")

        Actuals = col_namer.convert_to_pyspark_cols(Actuals)
        Time = col_namer.convert_to_pyspark_cols(Time).select(*time_dim_cols)

        logger.debug(f"Datatypes of the Time grain:{Time.dtypes}")

        # check the datatype of the keys column before joining
        actual_day_dtype = [dtype for name, dtype in Actuals.dtypes if name == time_join_key][0]
        dim_time_day_dtype = [dtype for name, dtype in Time.dtypes if name == time_join_key][0]
        assert (
            actual_day_dtype == dim_time_day_dtype
        ), f"Data Type mistmatch of the key columns: Actuals.[Day] {actual_day_dtype} and Time.[Day] {dim_time_day_dtype}"

        # Actuals Left Join of Time dimension to get the PW value
        combined_df = Actuals.join(Time, on=time_join_key, how="left")
        combined_df.persist(StorageLevel.MEMORY_AND_DISK)

        # Set the correct level to group the data
        actual_grains_new = [i for i in actual_grains_cols if i != time_join_key]
        logger.debug(f"actual_grains_new : {actual_grains_new}")
        group_cols_pw = actual_grains_new + [required_time_level]
        output_measures = set(Actuals.columns) - set(actual_grains_new)

        #  Group the actuals at PW level
        logger.debug(list(output_measures))
        logger.debug(f"Group by level grains are : {group_cols_pw}")

        # Create a dictionary which has all the agg func corresponding to the columns
        all_expr = {x: "sum" if x != time_join_key else "min" for x in output_measures}
        logger.debug(f"sum_expression : {all_expr}")
        Output = combined_df.groupby(*group_cols_pw).agg(all_expr)

        # Create the output PW measures
        output_measures_pw = [
            str(cols) + " PW" if cols != time_join_key else str(cols) for cols in output_measures
        ]

        rename_mapping = dict(
            zip(
                [f"sum({x})" if x != time_join_key else f"min({x})" for x in output_measures],
                output_measures_pw,
            )
        )
        logger.debug(f"rename_mapping values are : {rename_mapping}")
        Output = Output.drop(*[required_time_level])

        # Rename the grouped columns columns
        for old_name, new_name in rename_mapping.items():
            Output = Output.withColumnRenamed(old_name, new_name)

        Output = col_namer.convert_to_o9_cols(df=Output)

        logger.info("Output created... Returning it to the caller fn")
    except Exception as e:
        logger.exception(e)
        Output = None
    return Output
