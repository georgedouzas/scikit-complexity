"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseInteraction
from ._coupling import CouplingInteraction
from ._hyperedge import HyperedgeInteraction
from ._imitation import ImitationInteraction
from ._interaction import Interaction
from ._neighbor import NeighborInteraction
from ._pair import PairInteraction
from ._pair_potential import PairPotentialInteraction
from ._potential import PotentialInteraction
from ._spreading import SpreadingInteraction
from ._threshold import ThresholdInteraction

__all__: list[str] = [
    'BaseInteraction',
    'CouplingInteraction',
    'HyperedgeInteraction',
    'ImitationInteraction',
    'Interaction',
    'NeighborInteraction',
    'PairInteraction',
    'PairPotentialInteraction',
    'PotentialInteraction',
    'SpreadingInteraction',
    'ThresholdInteraction',
]
