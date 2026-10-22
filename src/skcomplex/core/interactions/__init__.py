"""Interactions that couple components through roles, over members, pairs, neighbours, or hyperedges."""

from ._interactions import Interaction
from ._matching import Match, matches, random_pairing
from ._potentials import PairPotential, Potential
from ._rewriting import Rewrite, Rewriting

__all__: list[str] = [
    'Interaction',
    'Match',
    'PairPotential',
    'Potential',
    'Rewrite',
    'Rewriting',
    'matches',
    'random_pairing',
]
