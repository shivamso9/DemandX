"""
Plugin : DP090PythonInfraValidation
Version : 2024.12
Maintained by : dpref@o9solutions.com

Script Params:
    ref_model_version: 2024.12
    required_libraries : "sktime, tslearn"

Input Queries:
    VersionMasterData : Select ([Version].[Version Name]) on row, () on column;


Output Variables:
    Status_output

Slice Dimension Attributes:

Pseudocode :


"""

import logging

from helpers.utils_pip_upgradation import upgrade_package

upgrade_package("statsforecast", "1.7.8")

logger = logging.getLogger("o9_logger")
import threading

import pandas as pd
from o9_common_utils.O9DataLake import O9DataLake
from packaging import version


def main(
    ref_model_version,
    VersionMasterData,
    df_keys={},
):

    # Creating empty dataframes

    Status_output = pd.DataFrame()

    try:
        # Create an empty list to store test results
        data = []

        # parsing the ref model version
        ref_model_version = version.parse(ref_model_version)

        ############# Python infrastructure setup in the tenant and is it of the appropriate version #############################

        try:
            import sys

            import sktime

            data.append({"Property": "Custom Conda Installation", "Status": "Success"})
        except ImportError:
            data.append({"Property": "Custom Conda Installation", "Status": "Failed"})

        ############################ Checking helpers and o9Reference Packages ################################################

        # try and catch block whether o9Reference is installed or not
        try:
            import o9Reference

            o9Reference_version = o9Reference.__version__
            o9Reference_version = version.parse(o9Reference_version)
            o9_reference_status = (
                "Success ({})".format(o9Reference_version)
                if (
                    o9Reference_version >= ref_model_version
                    or o9Reference_version == version.parse("0.0.0")
                )
                else "Failed ({})".format(o9Reference_version)
            )
            data.append(
                {
                    "Property": "DP Ref o9Reference Version",
                    "Status": o9_reference_status,
                }
            )
        except:
            data.append({"Property": "DP Ref o9Reference Version", "Status": "Missing"})

        try:
            import helpers

            o9helpers_version = helpers.__version__
            o9helpers_version = version.parse(o9helpers_version)
            o9_helpers_status = (
                "Success ({})".format(o9helpers_version)
                if (
                    o9helpers_version >= ref_model_version
                    or o9helpers_version == version.parse("0.0.0")
                )
                else "Failed ({})".format(o9helpers_version)
            )
            data.append(
                {
                    "Property": "DP Ref Helpers Version",
                    "Status": o9_helpers_status,
                }
            )
        except:
            data.append({"Property": "DP Ref Helpers Version", "Status": "Missing"})

        try:
            import cml

            data.append({"Property": "CML Version", "Status": "Success"})
        except:
            data.append({"Property": "CML Version", "Status": "Failed"})

        ############################ Checking Plugin read and write #############################################################

        # Checking whether plugin can read the data (VersionMasterData)
        try:
            if len(VersionMasterData) > 0:
                reading_inputs_status = "Success"
            else:
                reading_inputs_status = "Failed"
            data.append(
                {
                    "Property": "Reading from LS",
                    "Status": reading_inputs_status,
                }
            )
        except:
            data.append({"Property": "Reading from LS", "Status": "Failed"})

        # Convert the list of dictionaries to a DataFrame
        Status_temp = pd.DataFrame(data)

        json_data = Status_temp.to_json()

        # Creating a dataframe with version and strore json_data in a measure

        version_col = VersionMasterData.columns[0]
        version_name = VersionMasterData[version_col][0]

        Status_output = pd.DataFrame(
            {version_col: version_name, "Infra Validation Status": [json_data]}
        )

        logger.info("##############################################################")
        logger.info("Status_output : {}".format(Status_output))

    except Exception as e:
        logger.exception(f"Exception {e} for slice : {df_keys}")
        Status_output = pd.DataFrame()
        return Status_output

    return Status_output


# Function Calls

VersionMasterData = O9DataLake.get("VersionMasterData")


# Check if slicing variable is present
if "df_keys" not in locals():
    logger.info("No slicing configured, assigning empty dict to df_keys ...")
    df_keys = {}

logger.info("Slice : {}".format(df_keys))

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.

back_thread = threading.Thread(
    # target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

Status_output = main(
    ref_model_version=ref_model_version,
    VersionMasterData=VersionMasterData,
    df_keys=df_keys,
)

O9DataLake.put("Status_output", Status_output)
