"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseInteraction
from ._foraging import ForagingInteraction
from ._hyperedge import HyperedgeInteraction
from ._interaction import Interaction
from ._neighbor import NeighborInteraction
from ._pair import PairInteraction
from ._pair_potential import PairPotentialInteraction
from ._potential import PotentialInteraction
from ._regrowth import RegrowthInteraction
from ._random_walk import RandomWalkInteraction

__all__: list[str] = [
    'BaseInteraction',
    'ForagingInteraction',
    'HyperedgeInteraction',
    'Interaction',
    'NeighborInteraction',
    'PairInteraction',
    'PairPotentialInteraction',
    'PotentialInteraction',
    'RandomWalkInteraction',
    'RegrowthInteraction',
]
