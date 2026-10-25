"""Natural resources: harvesting, extracting, and cultivating renewable and nonrenewable stocks."""

from ._commons import (
    CowComponent,
    GrazingInteraction,
    HerderComponent,
    PastureComponent,
    RegulatorInteraction,
    richest_pasture,
    tragedy_of_the_commons,
)
from ._extraction import FisheryComponent, OilEconomyComponent, fishery_capital, oil_economy
from ._fisheries import hilsa_fishery
from ._food import food_security
from ._harvesting import FarmerComponent, HarvestInteraction, StockComponent, harvesting
from ._optimal_harvesting import HarvestedResourceComponent, optimal_harvesting
from ._plantations import cocoa

__all__: list[str] = [
    'CowComponent',
    'FarmerComponent',
    'FisheryComponent',
    'GrazingInteraction',
    'HarvestInteraction',
    'HarvestedResourceComponent',
    'HerderComponent',
    'OilEconomyComponent',
    'PastureComponent',
    'RegulatorInteraction',
    'StockComponent',
    'cocoa',
    'fishery_capital',
    'food_security',
    'harvesting',
    'hilsa_fishery',
    'oil_economy',
    'optimal_harvesting',
    'richest_pasture',
    'tragedy_of_the_commons',
]
