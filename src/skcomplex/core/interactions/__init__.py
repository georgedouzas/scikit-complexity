"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._alignment import AlignmentInteraction
from ._base import BaseInteraction
from ._choice import ChoiceInteraction
from ._evolution import EvolutionInteraction
from ._foraging import ForagingInteraction
from ._growth import GrowthInteraction
from ._hyperedge import HyperedgeInteraction
from ._interaction import Interaction
from ._neighbor import NeighborInteraction
from ._pair import PairInteraction
from ._pair_potential import PairPotentialInteraction
from ._potential import PotentialInteraction
from ._random_walk import RandomWalkInteraction
from ._reaction import ReactionInteraction
from ._regrowth import RegrowthInteraction

__all__: list[str] = [
    'AlignmentInteraction',
    'BaseInteraction',
    'ChoiceInteraction',
    'EvolutionInteraction',
    'ForagingInteraction',
    'GrowthInteraction',
    'HyperedgeInteraction',
    'Interaction',
    'NeighborInteraction',
    'PairInteraction',
    'PairPotentialInteraction',
    'PotentialInteraction',
    'RandomWalkInteraction',
    'ReactionInteraction',
    'RegrowthInteraction',
]
