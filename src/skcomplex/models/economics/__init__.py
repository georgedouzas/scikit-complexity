"""Economics: Solow growth in continuous time and the cobweb market in discrete time."""

from ._factories import Income, cobweb, multiplier_accelerator, solow
from ._growth import Cobweb, Economy, Market, SolowGrowth

__all__: list[str] = [
    'Cobweb',
    'Economy',
    'Income',
    'Market',
    'SolowGrowth',
    'cobweb',
    'multiplier_accelerator',
    'solow',
]
