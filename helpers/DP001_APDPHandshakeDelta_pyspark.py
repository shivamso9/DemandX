import logging

from o9Reference.spark_utils.common_utils import ColumnNamer
from pyspark.sql import functions as F

from helpers.pyspark.find_new_and_modified_record import find_new_and_modified_records

logger = logging.getLogger("o9_logger")
col_namer = ColumnNamer()


def main(
    AssortmentPlanning,
    OutputDataframe,
    DateGrains,
    DPIncrementalMeasure,
    PnlMaster,
    RegionMaster,
    AccountMaster,
    spark,
):
    try:
        version_col = "Version_VersionName"
        sales_planning_channel_col = "SalesDomain_SalesPlanningChannel"
        selling_season_col = "SellingSeason_SellingSeason"
        pl_item_col = "Item_PlanningItem"
        location_col = "Location_Location"
        pl_location_col = "Location_PlanningLocation"
        planning_channel_col = "Channel_PlanningChannel"
        planning_demand_domain_col = "DemandDomain_PlanningDemandDomain"
        start_week_sc = "Start Week SC L"
        end_week_sc = "End Week SC L"
        intro_date = "Intro Date"
        disco_date = "Disco Date"
        account = "Account_PlanningAccount"
        pnl = "PnL_PlanningPnL"
        region = "Region_PlanningRegion"

        outputmap = [
            version_col,
            planning_channel_col,
            pl_item_col,
            pl_location_col,
            planning_demand_domain_col,
            account,
            pnl,
            region,
            intro_date,
            disco_date,
            DPIncrementalMeasure,
        ]

        logger.info("Extracting dimension cols ...")
        input_dataframes = [
            (AssortmentPlanning, "AssortmentPlanning"),
            (PnlMaster, "PnlMaster"),
            (RegionMaster, "RegionMaster"),
            (AccountMaster, "AccountMaster"),
        ]

        for df, name in input_dataframes:
            if df.count() == 0:
                logger.error(f"{name} input is empty. Exiting without execution.")
                return spark.createDataFrame([], OutputDataframe.schema)

        AssortmentPlanning = col_namer.convert_to_pyspark_cols(AssortmentPlanning)
        OutputDataframe = col_namer.convert_to_pyspark_cols(OutputDataframe)
        required_columns = OutputDataframe.columns + [DPIncrementalMeasure]
        req_cols = [
            version_col,
            sales_planning_channel_col,
            pl_item_col,
            location_col,
            selling_season_col,
            start_week_sc,
            end_week_sc,
        ]
        AssortmentPlanning = AssortmentPlanning.select(*req_cols).dropDuplicates()
        col_mapping = {
            sales_planning_channel_col: planning_channel_col,
            selling_season_col: planning_demand_domain_col,
            start_week_sc: intro_date,
            end_week_sc: disco_date,
            location_col: pl_location_col,
        }
        for old_col, new_col in col_mapping.items():
            AssortmentPlanning = AssortmentPlanning.withColumnRenamed(old_col, new_col)
        common_cols = AssortmentPlanning.columns
        required_columns = common_cols + [DPIncrementalMeasure]
        OutputDataframe = OutputDataframe.select(*required_columns)
        AssortmentPlanning = col_namer.convert_to_o9_cols(AssortmentPlanning)
        OutputDataframe = col_namer.convert_to_o9_cols(OutputDataframe)
        filtered_df = find_new_and_modified_records(
            AssortmentPlanning, OutputDataframe, DateGrains, DPIncrementalMeasure
        )
        if filtered_df is None or filtered_df.count() == 0:
            logger.warning(
                "find_new_and_modified_records returned empty. Returning empty DataFrame."
            )
            return spark.createDataFrame([], OutputDataframe.schema)
        filtered_df = col_namer.convert_to_pyspark_cols(filtered_df)
        pnl_value = PnlMaster.first()[0]
        account_value = AccountMaster.first()[0]
        region_value = RegionMaster.first()[0]
        filtered_df = (
            filtered_df.withColumn(account, F.lit(account_value))
            .withColumn(pnl, F.lit(pnl_value))
            .withColumn(region, F.lit(region_value))
        )
        filtered_df = filtered_df.select(*outputmap)
        filtered_df = col_namer.convert_to_o9_cols(filtered_df)
        logger.info("successfully executed updating assortment batch ")
    except Exception as e:
        logger.exception(f"Exception for slice : {e}")
        try:
            return spark.createDataFrame([], OutputDataframe.schema)
        except Exception as inner_e:
            logger.exception(f"Fallback schema creation failed: {inner_e}")
            return spark.createDataFrame([], schema=None)
    return filtered_df
