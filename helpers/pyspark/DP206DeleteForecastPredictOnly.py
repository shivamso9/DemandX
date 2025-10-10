import logging

from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql.functions import col, lit
from pyspark.sql.types import DoubleType

logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


class Constants:
    version_name = "Version_VersionName"
    planning_item = "Item_PlanningItem"
    planning_Account = "Account_PlanningAccount"
    planning_channel = "Channel_PlanningChannel"
    planning_region = "Region_PlanningRegion"
    planning_pnl = "PnL_PlanningPnL"
    planning_location = "Location_PlanningLocation"
    planning_demand_domain = "DemandDomain_PlanningDemandDomain"
    assortment_batch = "DP Incremental Assortment Batch"
    promotion_batch = "DP Incremental Promotions Batch"
    ml_fcst_hml = "ML Fcst HML"

    JOIN_COLUMNS = [
        version_name,
        planning_item,
        planning_Account,
        planning_channel,
        planning_region,
        planning_pnl,
        planning_location,
        planning_demand_domain,
    ]


def main(Assortment, Forecast, spark):
    try:
        input_dataframes = [(Assortment, "Assortment"), (Forecast, "Forecast")]

        for df, name in input_dataframes:
            if df is None or df.rdd.isEmpty():
                logger.error(f"{name} input is empty. Exiting without execution.")
                return spark.createDataFrame([], schema=Forecast.schema)

        Assortment = col_namer.convert_to_pyspark_cols(Assortment)
        Forecast = col_namer.convert_to_pyspark_cols(Forecast)
        filtered_assortment = Assortment.filter(
            (col(Constants.assortment_batch) == 1) | (col(Constants.promotion_batch) == 1)
        )
        result = filtered_assortment.join(Forecast, Constants.JOIN_COLUMNS, "inner")
        final_result = result.select(Forecast.columns)
        final_result = final_result.withColumn(Constants.ml_fcst_hml, lit(None).cast(DoubleType()))
        final_result = col_namer.convert_to_o9_cols(final_result)
        return final_result
    except Exception as e:
        logger.exception(f"Exception for slice : {e}")
        try:
            return spark.createDataFrame([], Forecast.schema)
        except Exception as inner_e:
            logger.exception(f"Fallback schema creation failed: {inner_e}")

            return spark.createDataFrame([], schema=None)
