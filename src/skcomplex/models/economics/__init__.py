"""Economics: Solow growth in continuous time and the cobweb market in discrete time."""

from ._factories import cobweb, solow
from ._growth import Cobweb, Economy, Market, SolowGrowth

__all__: list[str] = [
    'Cobweb',
    'Economy',
    'Market',
    'SolowGrowth',
    'cobweb',
    'solow',
]
