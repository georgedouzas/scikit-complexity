"""Natural resources: harvesting, extracting, and cultivating renewable and nonrenewable stocks."""

from ._extraction import Fishery, OilEconomy, fishery_capital, oil_economy
from ._fisheries import hilsa_fishery
from ._food import food_security
from ._harvesting import Farmer, Harvest, Stock, harvesting
from ._plantations import cocoa

__all__: list[str] = [
    'Farmer',
    'Fishery',
    'Harvest',
    'OilEconomy',
    'Stock',
    'cocoa',
    'fishery_capital',
    'food_security',
    'harvesting',
    'hilsa_fishery',
    'oil_economy',
]
