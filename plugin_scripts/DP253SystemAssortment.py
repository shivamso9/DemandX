"""
Plugin : DP053SystemAssortment
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    ReadFromHive : False
    LocationColumnNames : Location.[All Location],Location.[Location Region],Location.[Stat Location],Location.[Location],Location.[Latitude],Location.[Location Status],Location.[New Store Flag],Location.[Store Intro Date],Location.[Store Disc Date],Location.[LAP1],Location.[LAP2],Location.[LAP3],Location.[LAP4],Location.[LAP5],Location.[LAP6],Location.[LAP7],Location.[LAP8],Location.[LAP9],Location.[LAP10],Location.[LAP11],Location.[LAP12],Location.[Location Country],Location.[Location City],Location.[Location Type],Location.[Location Zip],Location.[LA1],Location.[LA2],Location.[LA3],Location.[LA4],Location.[LA5],Location.[LA6],Location.[LA7],Location.[LA8]
    FeatureLevel : [Location].[Location]

Input Queries:
    LocationAttribute : Select (Location.[All Location]*Location.[Location]*Location.[Location City]*Location.[Location Country]*Location.[Location Region]*Location.[Location Type]*Location.[Location Zip]*Location.[Planning Location]*Location.[Reporting Location]*Location.[Stat Location]* Location.[LA1]* Location.[LA2]* Location.[LA3]* Location.[LA4]*Location.[LA5]*Location.[LA6]*Location.[LA7]*Location.[LA8]) on row,() on column include memberproperties {Location.[Location],[New Store Approve Flag]}{Location.[Location],[New Store Flag]}{Location.[Location],[Store Setup Complete]}{Location.[Location],[Store Intro Date]}{Location.[Location],[Store Disc Date]}{Location.[Location],[Address]}{Location.[Location],[Location Status]}{Location.[Location],[Latitude]}{Location.[Location],[Longitude]}{Location.[Location],[LAP1]}{Location.[Location],[LAP2]}{Location.[Location],[LAP3]}{Location.[Location],[LAP4]}{Location.[Location],[LAP5]}{Location.[Location],[LAP6]}{Location.[Location],[LAP7]}{Location.[Location],[LAP8]}{Location.[Location],[LAP9]}{Location.[Location],[LAP10]}{Location.[Location],[LAP11]}{Location.[Location],[LAP12]} INCLUDE_NULLMEMBERS;

    CurrentDay : select ([Version].[Version Name] * &CurrentDay);

    ApproveFlag : Select ([Version].[Version Name] * [Location].[Location] ) on row,  ({Measure.[New Store Approval Flag]}) on column;

    FinalLikeStore : (Select ([Version].[Version Name] * [Location].[Location] ) on row,  ({Measure.[Final Selected Like Store]}) on column).filter(~isnull(Measure.[Override Like Store Associations]));

    ExternalAssortment : Select ([Version].[Version Name] * [Region].[Planning Region] * [Item].[Item] * [PnL].[Planning PnL] * [Location].[Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] ) on row,  ({Measure.[External Assortment]}) on column;

    AssortmentFinal : Select ([Version].[Version Name] * [Region].[Planning Region] * [Item].[Item] * [PnL].[Planning PnL] * [Location].[Location] * [Demand Domain].[Planning Demand Domain] * [Account].[Planning Account] * [Channel].[Planning Channel] ) on row, ({Measure.[Assortment Final]}) on column;

    NewStoreAssortmentFlag : Select (FROM.[Item].[Item] * FROM.[Account].[Planning Account] * FROM.[Channel].[Planning Channel] * FROM.[Region].[Planning Region] * FROM.[PnL].[Planning PnL] * FROM.[Location].[Location] * FROM.[Demand Domain].[Planning Demand Domain] * TO.[Item].[Item] * TO.[Account].[Planning Account] * TO.[Channel].[Planning Channel] * TO.[Region].[Planning Region] * TO.[PnL].[Planning PnL] * TO.[Location].[Location] * TO.[Demand Domain].[Planning Demand Domain]) on row, ({Edge.[641 New Store Assortment].[New Store Assortment Flag]}) on column where {RelationshipType.[641 New Store Assortment], Version.[Version Name]};

    OverrideLikeStoreAssociation : Select ([Version].[Version Name] * [Location].[Location] ) on row, ({Measure.[Override Like Store Association]}) on column;

Output Variables:
    Result {{dummy}}

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

from ds_ref_lib.plugin_helpers.nsi.assortment.multiple_groups import main
from o9_common_utils.O9DataLake import O9DataLake

logger = logging.getLogger("o9_logger")

dummy = None

# Import Data ------------------------------------------------#
Location = O9DataLake.get("LocationAttribute")
ApproveFlag = O9DataLake.get("ApproveFlag")
CurrentDay = O9DataLake.get("CurrentDay")
FinalLikeStore = O9DataLake.get("FinalLikeStore")
Assortment = O9DataLake.get("AssortmentFinal")
ExternalAssortment = O9DataLake.get("ExternalAssortment")
NewStoreAssortmentFlag = O9DataLake.get("NewStoreAssortmentFlag")

# Calling the main function -----------------------------------#
try:
    Result = main(
        assortment=Assortment,
        location=Location,
        approve_flag=ApproveFlag,
        current_day=CurrentDay,
        like_store=FinalLikeStore,
        external=ExternalAssortment,
        new_store_assortment_flag=NewStoreAssortmentFlag,
        logger=logger,
    )

except Exception as e:
    logger.error(f"Exception for slice: {df_keys}")
    logger.exception(e)
    Result = pd.DataFrame()

finally:
    # Export output ------------------------------------------------#
    O9DataLake.put("dummy", Result)
