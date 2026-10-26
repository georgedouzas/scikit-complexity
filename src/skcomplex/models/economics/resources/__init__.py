"""Natural resources: harvesting, extracting, and cultivating renewable and nonrenewable stocks."""

from ._commons import (
    CowElement,
    GrazingInteraction,
    HerderElement,
    PastureElement,
    RegulatorInteraction,
    richest_pasture,
    tragedy_of_the_commons,
)
from ._extraction import FisheryElement, OilEconomyElement, fishery_capital, oil_economy
from ._fisheries import hilsa_fishery
from ._food import food_security
from ._harvesting import FarmerElement, HarvestInteraction, StockElement, harvesting
from ._optimal_harvesting import HarvestedResourceElement, optimal_harvesting
from ._plantations import cocoa

__all__: list[str] = [
    'CowElement',
    'FarmerElement',
    'FisheryElement',
    'GrazingInteraction',
    'HarvestInteraction',
    'HarvestedResourceElement',
    'HerderElement',
    'OilEconomyElement',
    'PastureElement',
    'RegulatorInteraction',
    'StockElement',
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
