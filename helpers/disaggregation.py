"""
Version : 0.0.0
Maintained by : dpref@o9solutions.com
"""

import logging

import pandas as pd

logger = logging.getLogger("o9_logger")


def join_lowest_level(
    df: pd.DataFrame,
    required_level: list,
    lowest_level: list,
    dim_master_data: list,
    is_lowest: list,
    join_on_lower_level: bool,
):
    assert (
        len(required_level) == len(lowest_level) == len(dim_master_data) == len(is_lowest)
    ), "all four list parameters should have equal length ..."

    if len(df) == 0:
        logger.warning("Empty dataframe provided ...")
        return df

    data = df.copy()
    for the_index, the_level in enumerate(required_level):
        logger.info("--- the_level : {}".format(the_level))
        if not is_lowest[the_index]:
            dim_df = dim_master_data[the_index][
                [required_level[the_index], lowest_level[the_index]]
            ].drop_duplicates()

            if join_on_lower_level:
                data = data.merge(dim_df, on=lowest_level[the_index], how="left")
            else:
                data = data.merge(dim_df, on=required_level[the_index], how="left")

    return data


def disaggregate_to_lower_level(
    df: pd.DataFrame,
    higher_level: list,
    base_measure: str,
    measures_to_disaggregate: list,
):
    data = df.copy()
    data["count"] = data.groupby(higher_level)[base_measure].transform("count")

    for the_col in measures_to_disaggregate:
        data[the_col] = data[the_col] / data["count"]

    return data
