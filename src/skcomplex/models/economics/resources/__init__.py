"""Natural resources: harvesting, extracting, and cultivating renewable and nonrenewable stocks."""

from ._commons import Cow, Grazing, Herder, Pasture, Regulator, RichestPasture, tragedy_of_the_commons
from ._extraction import Fishery, OilEconomy, fishery_capital, oil_economy
from ._fisheries import hilsa_fishery
from ._food import food_security
from ._harvesting import Farmer, Harvest, Stock, harvesting
from ._optimal_harvesting import HarvestedResource, optimal_harvesting
from ._plantations import cocoa

__all__: list[str] = [
    'Cow',
    'Farmer',
    'Fishery',
    'Grazing',
    'Harvest',
    'HarvestedResource',
    'Herder',
    'OilEconomy',
    'Pasture',
    'Regulator',
    'RichestPasture',
    'Stock',
    'cocoa',
    'fishery_capital',
    'food_security',
    'harvesting',
    'hilsa_fishery',
    'oil_economy',
    'optimal_harvesting',
    'tragedy_of_the_commons',
]
