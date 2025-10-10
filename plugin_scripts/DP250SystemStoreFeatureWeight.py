"""
Plugin : DP050SystemStoreFeatureWeight
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    mandatory_features : None
    remove_multicollinearity : False
    frequency : daily
    ReadFromHive : False
    numerical_cols : None
    LocationColumnNames : Location.[All Location],Location.[Location Region],Location.[Stat Location],Location.[Location],Location.[Latitude],Location.[Location Status],Location.[New Store Flag],Location.[Store Intro Date],Location.[Store Disc Date],Location.[LAP1],Location.[LAP2],Location.[LAP3],Location.[LAP4],Location.[LAP5],Location.[LAP6],Location.[LAP7],Location.[LAP8],Location.[LAP9],Location.[LAP10],Location.[LAP11],Location.[LAP12],Location.[Location Country],Location.[Location City],Location.[Location Type],Location.[Location Zip],Location.[LA1],Location.[LA2],Location.[LA3],Location.[LA4],Location.[LA5],Location.[LA6],Location.[LA7],Location.[LA8]
    FeatureNames : Location.[All Location],Location.[Location Region],Location.[Stat Location],Location.[Reporting Location],Location.[Reporting Location$DisplayName],Location.[Location],Location.[Location$DisplayName],Location.[Address],Location.[Latitude],Location.[Location Image],Location.[Location Status],Location.[Longitude],Location.[Description],Location.[New Store Flag],Location.[New Store Approve Flag],Location.[Store Setup Complete],Location.[LAP1],Location.[LAP2],Location.[LAP3],Location.[Store Intro Date],Location.[Store Disc Date], Location.[LAP6], Location.[LAP7], Location.[LAP8], Location.[LAP9], Location.[Location Country],Location.[Location City],Location.[Location Type],Location.[Location Zip],Location.[LA1],Location.[LA2],Location.[LA3],Location.[LA4]

Input Queries:
    CurrentDay : select ([Version].[Version Name] * &CurrentDay);

    LocationAttribute : Select (Location.[All Location]*Location.[Location]*Location.[Location City]*Location.[Location Country]*Location.[Location Region]*Location.[Location Type]*Location.[Location Zip]*Location.[Planning Location]*Location.[Reporting Location]*Location.[Stat Location]* Location.[LA1]* Location.[LA2]* Location.[LA3]* Location.[LA4]*Location.[LA5]*Location.[LA6]*Location.[LA7]*Location.[LA8]) on row,() on column include memberproperties {Location.[Location],[New Store Approve Flag]}{Location.[Location],[New Store Flag]}{Location.[Location],[Store Setup Complete]}{Location.[Location],[Store Intro Date]}{Location.[Location],[Store Disc Date]}{Location.[Location],[Address]}{Location.[Location],[Location Status]}{Location.[Location],[Latitude]}{Location.[Location],[Longitude]}{Location.[Location],[LAP1]}{Location.[Location],[LAP2]}{Location.[Location],[LAP3]}{Location.[Location],[LAP4]}{Location.[Location],[LAP5]}{Location.[Location],[LAP6]}{Location.[Location],[LAP7]}{Location.[Location],[LAP8]}{Location.[Location],[LAP9]}{Location.[Location],[LAP10]}{Location.[Location],[LAP11]}{Location.[Location],[LAP12]} INCLUDE_NULLMEMBERS;

    Parameters : Select (Version.[Version Name]) on row,  ({ Measure.[Like Store Search History Measure],Measure.[Like Store Location Search Space],Measure.[Like Store Channel Search Space] ,Measure.[Like Store Attributes], Measure.[Like Store Geo Attributes],Measure.[Like Store Other Attributes],Measure.[Like Item History Period]}) on column;

    Sales : Select ([Location].[Location] * [Time].[Day] * [Version].[Version Name]) on row, ({Measure.[Actual], Measure.[BackOrders], Measure.[Billing], Measure.[Orders], Measure.[Shipments]}) on column;


Output Variables:
    FeatureStoreWeight

Slice Dimension Attributes:
    Version.[Version Name]

"""

import logging

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

from ds_ref_lib.plugin_helpers.nsi.feature_imp.multiple_groups import main
from o9_common_utils.O9DataLake import O9DataLake

logger = logging.getLogger("o9_logger")

FeatureStoreWeight = None

# Import Data ------------------------------------------------#
Location = O9DataLake.get("LocationAttribute")
Params = O9DataLake.get("Parameters")
Sales = O9DataLake.get("Sales")
CurrentDay = O9DataLake.get("CurrentDay")

# Calling the main function -----------------------------------#

try:
    FeatureStoreWeight = main(
        data=Sales,
        location=Location,
        params=Params,
        current_day=CurrentDay,
        numerical_cols=numerical_cols,
        freq=frequency,
        vif_flag=remove_multicollinearity,
        mandatory_features=mandatory_features,
        logger=logger,
    )

except Exception as e:
    logger.error(f"Exception for slice: {df_keys}")
    logger.exception(e)
    FeatureStoreWeight = pd.DataFrame()

finally:
    # Export output ------------------------------------------------#
    O9DataLake.put("FeatureStoreWeight", FeatureStoreWeight)
