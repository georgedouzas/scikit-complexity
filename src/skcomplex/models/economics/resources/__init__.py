"""Natural resources: harvesting and extracting renewable and nonrenewable stocks."""

from ._extraction import Fishery, OilEconomy, fishery_capital, oil_economy
from ._harvesting import Farmer, Harvest, Stock, harvesting

__all__: list[str] = [
    'Farmer',
    'Fishery',
    'Harvest',
    'OilEconomy',
    'Stock',
    'fishery_capital',
    'harvesting',
    'oil_economy',
]
