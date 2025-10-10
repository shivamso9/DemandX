import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.decorators import (
    map_output_columns_to_dtypes,  # type: ignore
)
from o9Reference.common_utils.decorators import convert_category_cols_to_str
from o9Reference.common_utils.function_logger import (
    log_inputs_and_outputs,  # type: ignore
)
from o9Reference.common_utils.function_timer import timed
from o9Reference.spark_utils.common_utils import ColumnNamer  # type: ignore

from helpers.o9PySparkConstants import o9Constants

logger = logging.getLogger("o9_logger")

col_namer = ColumnNamer()


# TODO : Fill this with output column list
col_mapping = {
    # "Channel.[Planning Channel]": str,
    # "PnL.[Planning PnL]": str,
    # "Region.[Planning Region]": str,
    # "Demand Domain.[Planning Demand Domain]": str,
    # "Account.[Planning Account]": str,
    # "Version.[Version Name]": str,
    # "Location.[Planning Location]": str,
    # "Time.[Partial Week]": "datetime64[ns]",
    # "Item.[Planning Item]": str,
    "Consensus Fcst": float,
    # "Stat Fcst L0": int,
    # "Planner Fcst": float,
}


@log_inputs_and_outputs
@timed
@map_output_columns_to_dtypes(col_to_dtype_mapping=col_mapping)
@convert_category_cols_to_str
def main(
    Input: pd.DataFrame,
) -> pd.DataFrame:

    plugin_name = "DP071ConsensusFcstOverrideRetail"
    logger.info("Executing {} ...".format(plugin_name))
    try:
        # get the Columns required from the o9 constants file
        PLANNER_FCST = o9Constants.PLANNER_FCST
        STAT_FCST_L0 = o9Constants.STAT_FCST_L0
        CONSENSUS_FCST = o9Constants.CONSENSUS_FCST

        logger.debug(Input.head())

        # check if the required columns are present in the input data
        cols = Input.columns
        if PLANNER_FCST not in cols:
            raise ValueError(f"{PLANNER_FCST} col is not present in the Input data")
        elif STAT_FCST_L0 not in cols:
            raise ValueError(f"{STAT_FCST_L0} col is not present in the Input data")

        Input[CONSENSUS_FCST] = np.where(
            Input[PLANNER_FCST].notnull(),
            Input[PLANNER_FCST],
            Input[STAT_FCST_L0],
        )

        return Input
    except Exception as e:
        raise e
