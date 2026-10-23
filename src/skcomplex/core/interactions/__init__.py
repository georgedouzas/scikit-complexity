"""Interactions that couple components through roles, over members, pairs, neighbours, or hyperedges."""

from ._interactions import Interaction
from ._matching import Match, matches, random_pairing

__all__: list[str] = [
    'Interaction',
    'Match',
    'matches',
    'random_pairing',
]
