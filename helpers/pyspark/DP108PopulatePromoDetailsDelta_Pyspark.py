from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql.functions import col, lit
from pyspark.sql.types import DoubleType

col_namer = ColumnNamer()


class Constants:
    planning_channel = "Channel_PlanningChannel"
    planning_region = "Region_PlanningRegion"
    planning_location = "Location_PlanningLocation"
    planning_pnl = "PnL_PlanningPnL"
    planning_account = "Account_PlanningAccount"
    planning_demand = "DemandDomain_PlanningDemandDomain"
    channel = "Channel_Channel"
    location = "Location_Location"
    account = "Account_Account"
    demand = "DemandDomain_DemandDomain"
    pnl = "PnL_PnL"
    region = "Region_Region"
    promo_batch = "DP Incremental Promo Batch"
    promotion_batch = "DP Incremental Promotions Batch"
    intro_date = "Intro Date"
    disco_date = "Disco Date"


def update_assortment(Assortment, Output, logger):
    logger.info("Updating Assortment intersection")
    Assortment = col_namer.convert_to_pyspark_cols(Assortment)
    required_columns = Assortment.columns
    if Constants.promo_batch in Output.columns:
        filtered_df = Output.filter(col(Constants.promo_batch) == 1.0)
        filtered_df = col_namer.convert_to_pyspark_cols(filtered_df)
        column_mapping = {
            Constants.channel: Constants.planning_channel,
            Constants.location: Constants.planning_location,
            Constants.account: Constants.planning_account,
            Constants.demand: Constants.planning_demand,
            Constants.pnl: Constants.planning_pnl,
            Constants.region: Constants.planning_region,
        }
        for old_col, new_col in column_mapping.items():
            filtered_df = filtered_df.withColumnRenamed(old_col, new_col)
        exclude_cols = [Constants.promotion_batch, Constants.intro_date, Constants.disco_date]
        join_keys = [col for col in Assortment.columns if col not in exclude_cols]
        df = Assortment.join(filtered_df, on=join_keys, how="inner")
        df = df.select(required_columns)
        df = df.withColumn(Constants.promotion_batch, lit(1))
        df = df.withColumn(
            Constants.promotion_batch, col(Constants.promotion_batch).cast(DoubleType())
        )
        df = df.select(join_keys + [Constants.promotion_batch])

        df = col_namer.convert_to_o9_cols(df)
        logger.info("Successfully executed Update assortment")
        return df
    else:
        return Assortment.limit(0)
