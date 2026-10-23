"""Interactions that couple components through roles, over members, pairs, neighbours, or hyperedges."""

from ._interactions import Interaction
from ._potentials import PairPotential, Potential

__all__: list[str] = [
    'Interaction',
    'PairPotential',
    'Potential',
]
