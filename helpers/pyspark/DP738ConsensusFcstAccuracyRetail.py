import logging
from datetime import timedelta

# import json
import pyspark

# from o9Reference.common_utils.decorators import map_output_columns_to_dtypes  # type: ignore
from o9Reference.common_utils.function_logger import (
    log_inputs_and_outputs,  # type: ignore
)
from o9Reference.spark_utils.common_utils import ColumnNamer  # type: ignore
from o9Reference.spark_utils.common_utils import (  # type: ignore
    get_clean_string,
    is_dimension,
)
from pyspark.sql.functions import abs, coalesce, col, date_format, isnull, lit
from pyspark.sql.functions import max as max_
from pyspark.sql.functions import min as min_
from pyspark.sql.functions import sum as sum_
from pyspark.sql.functions import to_date, when
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

from helpers.o9PySparkConstants import o9PySparkConstants
from helpers.pyspark.aggregation_template import main as aggmain
from helpers.utils import (
    check_duplicates,
    get_grains_and_measures_from_dataset,
    get_list_of_grains_from_string,
)

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()

# TODO : Fill this with output column list
# col_mapping = {}


@log_inputs_and_outputs
# @map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
def main(
    StatFcst: pyspark.sql.dataframe.DataFrame,
    Actual: pyspark.sql.dataframe.DataFrame,
    LagAssociation: pyspark.sql.dataframe.DataFrame,
    ActualPlanningLevelGrains: str,
    StatFcstGrains: str,
    StatFcstMeasures: str,
    StatFcstL4Grains: str,
    CurrentTimePeriod: pyspark.sql.dataframe.DataFrame,
    L4ColumnMapping: str,
    MasterDataDict: dict,
    NWeeks: str,
    Aggregation: str,
    spark,
):
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    """Function to calculate the accuracy and bias metrics for the forecasts.

    Args:
        StatFcst (pyspark.sql.dataframe.DataFrame): Stat Fcst dataset at Planning level
        Actual (pyspark.sql.dataframe.DataFrame): Actuals dataset
        LagAssociation (pyspark.sql.dataframe.DataFrame): Lag Assocation dataset
        ActualPlanningLevelGrains (str): Actual grains required at planning level
        StatFcstGrains (str): Stat Forecast Grains
        StatFcstMeasures (str): Stat Fcst Measures
        StatFcstL4Grains (str): Stat Fcst Grains at L4 level
        CurrentTimePeriod (pyspark.sql.dataframe.DataFrame): Current Time Week
        L4ColumnMapping (str): Column mapping string
        MasterDataDict (dict): Master Data dictionary required
        NWeeks (str): Number weeks required to filter the past n weeks data
        Aggregation (str): Aggregation type

    Returns:
        dict: dictionary of output datasets
    """
    plugin_name = "DP738ConsensusAccuracyCalculation_Pyspark"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        # o9 Constants used in the code
        WEEK = o9PySparkConstants.WEEK  # Time_Week
        PARTIAL_WEEK = o9PySparkConstants.PARTIAL_WEEK
        CONSENSUS_FCST = o9PySparkConstants.CONSENSUS_FCST
        CONSENSUS_FCST_W_LAG = o9PySparkConstants.CONSENSUS_FCST_W_LAG
        CONSENSUS_FCST_W_LAG_ERROR = o9PySparkConstants.CONSENSUS_FCST_W_LAG_ERROR
        CONSENSUS_FCST_W_LAG_ABS_ERROR = o9PySparkConstants.CONSENSUS_FCST_W_LAG_ABS_ERROR
        CONSENSUS_FCST_W_LAG_WMAPE_RETAIL = o9PySparkConstants.CONSENSUS_FCST_W_LAG_WMAPE_RETAIL
        CONSENSUS_FCST_W_LAG_ACCURACY_RETAIL = (
            o9PySparkConstants.CONSENSUS_FCST_W_LAG_ACCURACY_RETAIL
        )
        CONSENSUS_FCST_W_LAG_BIAS_RETAIL = o9PySparkConstants.CONSENSUS_FCST_W_LAG_BIAS_RETAIL
        CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_RETAIL = (
            o9PySparkConstants.CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_RETAIL
        )
        CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_INDICATOR_RETAIL = (
            o9PySparkConstants.CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_INDICATOR_RETAIL
        )
        CONSENSUS_FCST_L4_W_LAG_RETAIL = o9PySparkConstants.CONSENSUS_FCST_L4_W_LAG_RETAIL
        CONSENSUS_FCST_L4_W_LAG_ABS_ERROR_RETAIL = (
            o9PySparkConstants.CONSENSUS_FCST_L4_W_LAG_ABS_ERROR_RETAIL
        )
        ACTUAL = o9PySparkConstants.ACTUAL

        # Get the Time Master data from MasterDataDict
        TimeMaster = MasterDataDict["TimeMaster"]

        # Check if the input has values
        if Actual.count() == 0:
            raise ValueError("Actuals dataframe cannot be empty! Check the input data sources!")
        elif StatFcst.count() == 0:
            raise ValueError("Stat Fcst data cannot be empty! Check the input data sources!")
        elif LagAssociation.count() == 0:
            raise ValueError("Lag Association data cannot be empty! Check the input data sources!")
        elif TimeMaster.count() == 0:
            raise ValueError("Time Master data cannot be empty! Check the input data sources!")

        # Get grains and measures from the input columns
        # (
        #     stat_fcst_grains,
        #     stat_fcst_measures,
        # ) = get_grains_and_measures_from_dataset(list_of_cols=StatFcst.columns)
        StatFcst = col_namer.convert_to_pyspark_cols(StatFcst)

        (
            lag_association_grains,
            lag_association_measures,
        ) = get_grains_and_measures_from_dataset(list_of_cols=LagAssociation.columns)
        LagAssociation = col_namer.convert_to_pyspark_cols(LagAssociation)

        TimeMaster = col_namer.convert_to_pyspark_cols(TimeMaster)

        # convert the grains to pyspark format columns
        StatFcstGrains = get_list_of_grains_from_string(input=StatFcstGrains)
        StatFcstGrains = [get_clean_string(x) for x in StatFcstGrains]

        StatFcstMeasures = [StatFcstMeasures]

        StatFcstL4Grains = get_list_of_grains_from_string(input=StatFcstL4Grains)
        StatFcstL4Grains = [get_clean_string(x) for x in StatFcstL4Grains]

        logger.debug("Grains and Measures of different Inputs --- ")
        logger.debug(
            f"stat_fcst_grains : {StatFcstGrains}, stat_fcst_measures : {StatFcstMeasures}"
        )
        logger.debug(
            f"lag_association_grains : {lag_association_grains}, lag_association_measures : {lag_association_measures}"
        )
        # logger.debug(TimeGrains)

        # Aggregate the Actuals to Planning Level

        # logger.debug(Actual.limit(5).show())
        ActualGrains = ",".join([column for column in Actual.columns if is_dimension(column)])
        logger.debug(ActualGrains)
        Actual = aggmain(
            SourceMeasures=ACTUAL,
            SourceGrain=ActualGrains,
            TargetMeasures=ACTUAL,
            TargetGrain=ActualPlanningLevelGrains,
            Aggregation=Aggregation,
            Input=Actual,
            MasterDataDict=MasterDataDict,
        )
        # logger.debug(Actual.limit(5).show())

        # Actual.repartition(1).write.mode("overwrite").csv(
        #     "../data/AccuracyOutputs/ActualPlanningLevel.csv", header=True
        # )
        ##############################################################################
        actual_grains, actual_measures = get_grains_and_measures_from_dataset(
            list_of_cols=Actual.columns
        )
        Actual = col_namer.convert_to_pyspark_cols(Actual)

        Measures = StatFcstMeasures + actual_measures + lag_association_measures
        logger.debug(f"Measures : {Measures}")

        # dict for all the required inputs
        check_duplicates(StatFcst)
        data_dict = {
            "StatFcst": StatFcst,
            "Actual": Actual,
            "LagAssociation": LagAssociation,
        }

        # Iterate through the above dict and join the above datasets to create one Output dataset
        # to calculate different metrics using Acutals, Stat Fcst and Lag Association

        # Check if Time_Week is present in the Time Master dataset
        if WEEK not in TimeMaster.columns:
            raise ValueError(
                f"{WEEK} column is not present in TimeMaster dataset. Please check the inputs and try again!"
            )

        for idx, (name, data) in enumerate(data_dict.items()):

            logger.debug(f"idx : {idx}, name : {name}")
            common_cols = set(data.columns).intersection(TimeMaster.columns)
            logger.debug(f"common_cols : {common_cols}")

            if len(common_cols) == 0:
                raise ValueError(
                    f"No Common cols are present between {TimeMaster.columns} and {data.columns}"
                )

            time_cols = list(common_cols)
            time_cols.append(WEEK)
            logger.debug(f"time_cols : {time_cols}")

            # Get the data at Time.[Week] level
            check_duplicates(data)
            data = data.join(
                TimeMaster.select(*time_cols).dropDuplicates(),
                on=list(common_cols),
                how="inner",
            )
            check_duplicates(data)
            # Drop the columns that are not Time_Week but are other Time dimensions
            cols_to_drop = [
                column for column in data.columns if column.startswith("Time") and column != WEEK
            ]
            logger.debug(f"cols to drop : {cols_to_drop}")
            data = data.drop(*cols_to_drop)

            if name == "StatFcst":
                grouped_cols = StatFcstGrains
                output_measures = StatFcstMeasures
            elif name == "Actual":
                grouped_cols = actual_grains
                output_measures = actual_measures
            elif name == "LagAssociation":
                grouped_cols = lag_association_grains
                output_measures = lag_association_measures
            else:
                raise ValueError(
                    f"Given name : {name} and dataframe are not present in the dictionary!"
                )
            grouped_cols = list(set(grouped_cols) - set(cols_to_drop))
            grouped_cols = grouped_cols + [WEEK]

            # Group the data at Time Week Level
            logger.debug(f"grouped_cols: {grouped_cols}")
            logger.debug(f"output_measures : {output_measures}")
            all_expr = {
                x: "sum" if x not in lag_association_measures else "max" for x in output_measures
            }
            logger.debug(all_expr)
            data = data.groupby(*grouped_cols).agg(all_expr)

            # Renaming the aggregated columns for each inputs
            rename_mapping = dict(
                zip(
                    [
                        (f"sum({x})" if x not in lag_association_measures else f"max({x})")
                        for x in output_measures
                    ],
                    output_measures,
                )
            )
            logger.debug(f"rename_mapping : {rename_mapping}")

            for old_name, new_name in rename_mapping.items():
                data = data.withColumnRenamed(old_name, new_name)

            check_duplicates(data)
            # Overwrite the dictionary data values with the data at Time Week level
            # data_dict[name] = data

            if idx == 0:
                Output = data

            # Joining each of the inputs
            elif idx > 0:
                logger.debug(f"Joining {Output.columns} and {data.columns}")
                common_cols = set(Output.columns).intersection(data.columns)
                logger.debug(f"join_cols : {common_cols}")

                Output = data.join(Output, on=list(common_cols), how="inner")

            logger.debug(Output.count())
            if Output.count() == 0:
                raise ValueError(
                    "The above join didn't work please check the input data and common keys!"
                )
            logger.debug(f"{name} is aggregated to {WEEK} level! \n")

        # Required Grains and Measures
        StatFcstGrains = [x for x in StatFcstGrains if x not in PARTIAL_WEEK] + [WEEK]
        required_grains_measures = StatFcstGrains + Measures
        logger.debug(f"Required Grains and Measures for Stat Fcst : {required_grains_measures}")

        check_duplicates(Output)
        # Output.repartition(1).write.mode("overwrite").csv(
        #     "../data/AccuracyOutputs/InputTimeWeekData.csv", header=True
        # )
        # logger.debug("Written the joined data to local!! \n")
        # Calculation of different Accuracy and BIAS metrics to be used for Retail

        # ACTUAL = actual_measures[0]
        W_to_PW_Lag_Association = lag_association_measures[0]

        logger.debug(f"Check the count of Output before filtering for nweeks : {Output.count()}")
        # Calculate the Stat Fcst W Lag value
        OutputItem = Output.withColumn(
            CONSENSUS_FCST_W_LAG,
            when(
                col(W_to_PW_Lag_Association).isNotNull(),
                col(W_to_PW_Lag_Association) * col(CONSENSUS_FCST),
            ),
        )
        # logger.debug(fcst_with_lags_data.dtypes)

        OutputItem = OutputItem.withColumn(
            WEEK,
            to_date(col(WEEK), "dd-MMM-yy"),
        )
        logger.debug(OutputItem.dtypes)

        # filter last n weeks data
        n_weeks = int(NWeeks.strip())
        logger.debug(f"n_weeks : {n_weeks}")

        if n_weeks < 0:
            raise ValueError(
                "NWeeks input script parameter value should be positive! Please check the script params and re-enter the value!"
            )

        # Get the value for Current date
        current_date = (
            col_namer.convert_to_pyspark_cols(CurrentTimePeriod)
            .withColumn(
                WEEK,
                to_date(col(WEEK), "dd-MMM-yy"),
            )
            .select(WEEK)
            .collect()[0][0]
        )
        logger.debug(current_date)

        date_n_weeks_ago = current_date - timedelta(weeks=n_weeks)
        logger.debug(date_n_weeks_ago)

        # Get the range of dates present in the data
        min_date, max_date = OutputItem.select(min_(WEEK), max_(WEEK)).first()
        logger.debug(f"The min and max date for OutputItem is {min_date}, {max_date}")

        # Check if filter date is out of range
        if date_n_weeks_ago >= max_date:
            logger.debug(
                f"(Current Date - NWeeks) = {date_n_weeks_ago} cannot be greater than the max date {max_date} of {WEEK} col in the data!"
            )
            # raise ValueError(
            #     f"(Current Date - NWeeks) = {date_n_weeks_ago} cannot be greater than the max date {max_date} of {WEEK} col in the data!"
            # )
        else:
            OutputItem = OutputItem.filter(col(WEEK) >= date_n_weeks_ago)

        logger.debug(
            f"After filtering for last {n_weeks} the Output data count : {OutputItem.count()}"
        )
        # Calculate the Fcst W Lag Error
        OutputItem = OutputItem.withColumn(
            CONSENSUS_FCST_W_LAG_ERROR,
            when(col(ACTUAL) < 0, col(CONSENSUS_FCST_W_LAG))
            .when((col(CONSENSUS_FCST_W_LAG) == 0) & (col(ACTUAL) == 0), 0)
            .when(isnull(col(CONSENSUS_FCST_W_LAG)) & (col(ACTUAL) == 0), None)
            .when(isnull(col(ACTUAL)) & (col(CONSENSUS_FCST_W_LAG) == 0), None)
            .when(
                (isnull(col(ACTUAL))) & (isnull(col(CONSENSUS_FCST_W_LAG))),
                None,
            )
            .otherwise(coalesce(col(CONSENSUS_FCST_W_LAG), lit(0)) - coalesce(col(ACTUAL), lit(0))),
        )

        # Calculate the Fcst W Lags Abs Error
        OutputItem = OutputItem.withColumn(
            CONSENSUS_FCST_W_LAG_ABS_ERROR,
            abs(col(CONSENSUS_FCST_W_LAG_ERROR)),
        )

        # Calculate the stat fcst w lag mape
        OutputItem = OutputItem.withColumn(
            CONSENSUS_FCST_W_LAG_WMAPE_RETAIL,
            when(
                ~isnull(col(CONSENSUS_FCST_W_LAG_ABS_ERROR))
                & (col(CONSENSUS_FCST_W_LAG_ABS_ERROR) != 0),
                when(col(ACTUAL) == 0, 1).otherwise(
                    col(CONSENSUS_FCST_W_LAG_ABS_ERROR) / col(ACTUAL)
                ),
            ).otherwise(lit(None)),
        )

        # Calculate the stat fcst accuracy
        OutputItem = OutputItem.withColumn(
            CONSENSUS_FCST_W_LAG_ACCURACY_RETAIL,
            when(
                ~isnull(col(ACTUAL)),
                when((1 - col(CONSENSUS_FCST_W_LAG_WMAPE_RETAIL)) <= 0, 0)
                .when((1 - col(CONSENSUS_FCST_W_LAG_WMAPE_RETAIL)) >= 1, 1)
                .otherwise(1 - col(CONSENSUS_FCST_W_LAG_WMAPE_RETAIL)),
            ),
        )

        # Calculate the stat fcst w lag bias
        OutputItem = OutputItem.withColumn(
            CONSENSUS_FCST_W_LAG_BIAS_RETAIL,
            when(
                col(ACTUAL) != 0,
                col(CONSENSUS_FCST_W_LAG_ERROR) / col(ACTUAL),
            ).otherwise(lit(None)),
        )

        # Calculate the stat fcst w lag tracking signal
        OutputItem = OutputItem.withColumn(
            CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_INDICATOR_RETAIL,
            when(
                col(ACTUAL) != 0,
                when((col(CONSENSUS_FCST_W_LAG) / col(ACTUAL)) <= 0.75, -1)
                .when((col(CONSENSUS_FCST_W_LAG) / col(ACTUAL)) >= 1.25, 1)
                .otherwise(0),
            ),
        )

        # Calculate the stat fcst w lag tracking signal indicator
        partition_cols = [x for x in Output.columns if x not in Measures and x != WEEK]
        logger.debug(f"parition by cols : {partition_cols}")
        windowspec = Window.partitionBy(*partition_cols).orderBy(WEEK).rowsBetween(-5, -1)
        OutputItem = OutputItem.withColumn(
            CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_RETAIL,
            sum_(col(CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_INDICATOR_RETAIL)).over(windowspec),
        )

        # logger.debug(OutputItem.limit(5).show())

        # check if all the grains are present for Stat Fcst
        logger.debug(f"StatFcstGrains are : {StatFcstGrains}")
        grains_not_present = [
            column for column in StatFcstGrains if column not in OutputItem.columns
        ]
        if len(grains_not_present) > 0:
            raise ValueError(
                f"These cols are not present in the output stat fcst grains : {grains_not_present}"
            )

        OutputItem = OutputItem.drop(*Measures)
        OutputItem = OutputItem.withColumn(
            WEEK,
            date_format(to_date(col(WEEK), "dd-MMM-yy"), "dd-LLL-yy"),
        )
        OutputItem = OutputItem.withColumn(
            CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_RETAIL,
            col(CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_RETAIL).cast(DoubleType()),
        ).withColumn(
            CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_INDICATOR_RETAIL,
            col(CONSENSUS_FCST_W_LAG_TRACKING_SIGNAL_INDICATOR_RETAIL).cast(DoubleType()),
        )
        logger.debug(
            f"updated date data type : - {OutputItem.dtypes} and count : {OutputItem.count()}"
        )
        # OutputItem.repartition(1).write.mode("overwrite").csv(
        #     "../data/AccuracyOutputs/RetailAccuracyItemLevel.csv", header=True
        # )
        # logger.debug(f"written retail accuracy calculations at item level!")

        # Calculate all Inputs combined at Item L4 level
        logger.debug("Convert the Outputs to L4 level!")
        OutputL4 = Output
        # split on delimiter and obtain grains
        L4ColumnMapping = L4ColumnMapping.split(";")
        # remove leading/trailing spaces if any
        L4ColumnMapping = [x.strip() for x in L4ColumnMapping]
        L4_level_columns_mapping = {
            val.split(":")[0]: val.split(":")[-1].strip() for val in L4ColumnMapping
        }

        logger.debug("L4 Column Mapping - ")
        # L4_level_columns_mapping = json.loads(L4ColumnMapping)
        logger.debug(L4_level_columns_mapping)
        logger.debug(type(L4_level_columns_mapping))

        for name, value in L4_level_columns_mapping.items():
            master_data = MasterDataDict[name]
            master_data = col_namer.convert_to_pyspark_cols(master_data)
            master_data_columns = get_list_of_grains_from_string(input=value)
            master_data_columns = [get_clean_string(x) for x in master_data_columns]
            logger.debug(f"master_data_columns : {master_data_columns}")

            join_cols = list(set(Output.columns).intersection(master_data.columns))
            logger.debug(f"join cols : {join_cols}")

            if len(join_cols) == 0:
                raise ValueError(
                    f"No common columns present between planning level Output and {name} Master Data!"
                )
            OutputL4 = OutputL4.join(
                master_data.select(*master_data_columns).drop_duplicates(),
                on=join_cols,
                how="inner",
            )

        required_grains_measures = StatFcstL4Grains + Measures
        OutputL4 = OutputL4.select(*required_grains_measures)

        expr = {x: "max" if x in lag_association_measures else "sum" for x in Measures}
        logger.debug(f"group by cols : {StatFcstL4Grains}")
        OutputL4 = OutputL4.groupby(*StatFcstL4Grains).agg(expr)

        logger.debug(OutputL4.count())

        rename_mapping = dict(
            zip(
                [(f"max({x})" if x in lag_association_measures else f"sum({x})") for x in Measures],
                Measures,
            )
        )
        logger.debug(rename_mapping)

        # Rename the grouped columns columns
        for old_name, new_name in rename_mapping.items():
            OutputL4 = OutputL4.withColumnRenamed(old_name, new_name)

        # Calculate the new additional metrics
        OutputL4 = OutputL4.withColumn(
            CONSENSUS_FCST_L4_W_LAG_RETAIL,
            when(
                col(W_to_PW_Lag_Association).isNotNull(),
                col(W_to_PW_Lag_Association) * col(CONSENSUS_FCST),
            ),
        )

        # filter for last n weeks
        # OutputL4 = OutputL4.filter(
        #     col(WEEK) >= date_n_weeks_ago
        # )

        OutputL4 = OutputL4.withColumn(
            CONSENSUS_FCST_L4_W_LAG_ABS_ERROR_RETAIL,
            when(
                coalesce(col(ACTUAL), lit(0)) < 0,
                col(CONSENSUS_FCST_L4_W_LAG_RETAIL),
            )
            .when(
                isnull(col(CONSENSUS_FCST_L4_W_LAG_RETAIL)) & (col(ACTUAL) == 0),
                None,
            )
            .when(
                isnull(col(ACTUAL)) & (col(CONSENSUS_FCST_L4_W_LAG_RETAIL) == 0),
                None,
            )
            .otherwise(abs(col(CONSENSUS_FCST_L4_W_LAG_RETAIL) + (-1 * col(ACTUAL)))),
        )

        grains_not_present = [col for col in StatFcstL4Grains if col not in OutputL4.columns]
        if len(grains_not_present) > 0:
            raise ValueError(
                f"These cols are not present in the output stat fcst l4 grains : {grains_not_present}"
            )

        OutputL4 = OutputL4.drop(*Measures)
        # OutputL4.repartition(1).write.mode("overwrite").csv(
        #     "../data/AccuracyOutputs/RetailAccuracyItemL4Level.csv",
        #     header=True,
        # )

        logger.debug(f"Output Item cols : {OutputItem.columns}")

        # drop a column "Forecast" from OutputItem using pyspark and drop duplicates
        OutputItem = OutputItem.drop(o9PySparkConstants.FORECAST_ITERATION).drop_duplicates()
        OutputItem = col_namer.convert_to_o9_cols(df=OutputItem)
        OutputL4 = col_namer.convert_to_o9_cols(df=OutputL4)

        logger.debug(OutputItem.count())
        logger.debug(OutputL4.count())
        # logger.debug(ActualL4.count())
        return {
            "AccuracyItemLevel": OutputItem,
            "AccuracyItemL4Level": OutputL4,
        }

    except Exception as e:
        raise e
