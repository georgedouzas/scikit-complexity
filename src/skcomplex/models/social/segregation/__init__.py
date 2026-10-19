"""Segregation: residential sorting of agents."""

from ._blocks import BlockMoves, Dweller, stochastic_schelling
from ._schelling import Resident, schelling

__all__: list[str] = [
    'BlockMoves',
    'Dweller',
    'Resident',
    'schelling',
    'stochastic_schelling',
]
