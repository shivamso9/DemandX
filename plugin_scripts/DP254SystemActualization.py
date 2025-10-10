"""
Plugin : DP054SystemActualization
Version : 0.0.0
Maintained by : dpref@o9solutions.com

Script Params:
    ReadFromHive : False

Input Queries:
    CurrentDay : select ([Version].[Version Name] * &CurrentDay);

    ScalingFactor : Select ([Version].[Version Name] * [Item].[Item] * [Account].[Account] * [Channel].[Channel] * [Region].[Region] * [PnL].[PnL] * [Location].[Location] * [Demand Domain].[Demand Domain] ) on row, ({Measure.[NSI Scaling Factor]}) on column;

    AccountAttribute : Select ([Account].[Planning Account]*[Account].[Account]);

    ChannelAttribute : Select ([Channel].[Planning Channel]*[Channel].[Channel]);

    RegionAttribute : Select ([Region].[Planning Region]*[Region].[Region]);

    DemandDomainAttribute : Select ([Demand Domain].[Planning Demand Domain]*[Demand Domain].[Demand Domain]);

    PnLAttribute : Select ([PnL].[Planning PnL]*[PnL].[PnL]);

    LocationAttribute : Select (Location.[All Location]*Location.[Location]*Location.[Location City]*Location.[Location Country]*Location.[Location Region]*Location.[Location Type]*Location.[Location Zip]*Location.[Planning Location]*Location.[Reporting Location]*Location.[Stat Location]* Location.[LA1]* Location.[LA2]* Location.[LA3]* Location.[LA4]*Location.[LA5]*Location.[LA6]*Location.[LA7]*Location.[LA8]) on row,() on column include memberproperties {Location.[Location],[New Store Approve Flag]}{Location.[Location],[New Store Flag]}{Location.[Location],[Store Setup Complete]}{Location.[Location],[Store Intro Date]}{Location.[Location],[Store Disc Date]}{Location.[Location],[Address]}{Location.[Location],[Location Status]}{Location.[Location],[Latitude]}{Location.[Location],[Longitude]}{Location.[Location],[LAP1]}{Location.[Location],[LAP2]}{Location.[Location],[LAP3]}{Location.[Location],[LAP4]}{Location.[Location],[LAP5]}{Location.[Location],[LAP6]}{Location.[Location],[LAP7]}{Location.[Location],[LAP8]}{Location.[Location],[LAP9]}{Location.[Location],[LAP10]}{Location.[Location],[LAP11]}{Location.[Location],[LAP12]} INCLUDE_NULLMEMBERS;

    Sales : Select ([Version].[Version Name] * [Item].[Item] * [Account].[Account] * [Channel].[Channel] * [Region].[Region] * [PnL].[PnL] * [Location].[Location] * [Demand Domain].[Demand Domain]* [Time].[Day] ) on row,  ({Measure.[Actual],Measure.[Backorders],Measure.[Billing],Measure.[Orders],Measure.[Shipments]}) on column;

    NewStoreAssortmentFlag : (Select (FROM.[PnL].[Planning PnL] * FROM.[Item].[Item] * FROM.[Account].[Planning Account] * FROM.[Channel].[Planning Channel] * FROM.[Location].[Location] * FROM.[Region].[Planning Region] * FROM.[Demand Domain].[Planning Demand Domain] * TO.[Location].[Location] * TO.[Region].[Planning Region] * TO.[Demand Domain].[Planning Demand Domain] * TO.[Account].[Planning Account] * TO.[PnL].[Planning PnL] * TO.[Channel].[Planning Channel] * TO.[Item].[Item]) on row, ({Edge.[641 New Store Assortment].[New Store Assortment Flag] }) on column where {RelationshipType.[641 New Store Assortment], &CWVAndScenarios}).filter(~isnull(Edge.[641 New Store Assortment].[New Store Assortment Flag]));

    NewStoreActualizationStatus : Select (FROM.[PnL].[Planning PnL] * FROM.[Item].[Item] * FROM.[Account].[Planning Account] * FROM.[Channel].[Planning Channel] * FROM.[Location].[Location] * FROM.[Region].[Planning Region] * FROM.[Demand Domain].[Planning Demand Domain] * TO.[Location].[Location] * TO.[Region].[Planning Region] * TO.[Demand Domain].[Planning Demand Domain] * TO.[Account].[Planning Account] * TO.[PnL].[Planning PnL] * TO.[Channel].[Planning Channel] * TO.[Item].[Item]) on row, ({Edge.[641 New Store Assortment].[New Store Actualization Status]}) on column where {RelationshipType.[641 New Store Assortment], Version.[Version Name]};

    OverrideLikeStoreAssociation : Select ([Version].[Version Name] * [Location].[Location] ) on row,  ({Measure.[Override Like Store Association]}) on column;

Output Variables:
    Result1 {{dummy}}, Result2 {{dummy2}}

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

from ds_ref_lib.plugin_helpers.nsi.actualization.multiple_groups import main
from o9_common_utils.O9DataLake import O9DataLake

logger = logging.getLogger("o9_logger")

dummy = None
dummy2 = None

# Import Data ------------------------------------------------#
Actual = O9DataLake.get("Sales")
CurrentDay = O9DataLake.get("CurrentDay")
AccountAttribute = O9DataLake.get("AccountAttribute")
ChannelAttribute = O9DataLake.get("ChannelAttribute")
DemandDomainAttribute = O9DataLake.get("DemandDomainAttribute")
LocationAttribute = O9DataLake.get("LocationAttribute")
PnLAttribute = O9DataLake.get("PnLAttribute")
RegionAttribute = O9DataLake.get("RegionAttribute")
OverrideLikeStoreAssociation = O9DataLake.get("OverrideLikeStoreAssociation")
NewStoreActualizationStatus = O9DataLake.get("NewStoreActualizationStatus")
NewStoreAssortmentFlag = O9DataLake.get("NewStoreAssortmentFlag")
ScalingFactor = O9DataLake.get("ScalingFactor")

# Calling the main function -----------------------------------#
try:
    Result1, Result2 = main(
        actual=Actual,
        current_day=CurrentDay,
        new_store_actualization_status=NewStoreActualizationStatus,
        new_store_assortment_flag=NewStoreAssortmentFlag,
        scaling_factor=ScalingFactor,
        locationdim=LocationAttribute,
        channeldim=ChannelAttribute,
        pnldim=PnLAttribute,
        regiondim=RegionAttribute,
        accountdim=AccountAttribute,
        demanddomaindim=DemandDomainAttribute,
        override_association=OverrideLikeStoreAssociation,
        logger=logger,
    )

except Exception as e:
    logger.error(f"Exception for slice: {df_keys}")
    logger.exception(e)
    Result1 = pd.DataFrame()
    Result2 = pd.DataFrame()

finally:
    # Export output ------------------------------------------------#
    O9DataLake.put("dummy", Result1)
    O9DataLake.put("dummy2", Result2)
