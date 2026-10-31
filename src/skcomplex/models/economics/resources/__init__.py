"""Natural resources: harvesting, extracting, and cultivating renewable and nonrenewable stocks."""

from ._commons import (
    CowElement,
    GrazingInteraction,
    HerderElement,
    PastureElement,
    QuotaRegulatorInteraction,
    tragedy_of_the_commons,
)
from ._extraction import FisheryElement, OilEconomyElement, fishery_capital, oil_economy
from ._fisheries import HilsaElement, hilsa_fishery
from ._food import RiceSectorElement, food_security
from ._harvesting import FarmerElement, HarvestInteraction, StockElement, harvesting
from ._optimal_harvesting import HarvestedResourceElement, optimal_harvesting
from ._plantations import PlantationElement, cocoa

__all__: list[str] = [
    'CowElement',
    'FarmerElement',
    'FisheryElement',
    'GrazingInteraction',
    'HarvestInteraction',
    'HarvestedResourceElement',
    'HerderElement',
    'HilsaElement',
    'OilEconomyElement',
    'PastureElement',
    'PlantationElement',
    'QuotaRegulatorInteraction',
    'RiceSectorElement',
    'StockElement',
    'cocoa',
    'fishery_capital',
    'food_security',
    'harvesting',
    'hilsa_fishery',
    'oil_economy',
    'optimal_harvesting',
    'tragedy_of_the_commons',
]
