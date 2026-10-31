"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseInteraction
from ._choice import ChoiceInteraction
from ._evolution import EvolutionInteraction
from ._hyperedge import HyperedgeInteraction
from ._interaction import Interaction
from ._neighbor import NeighborInteraction
from ._pair import PairInteraction
from ._pair_potential import PairPotentialInteraction
from ._potential import PotentialInteraction

__all__: list[str] = [
    'BaseInteraction',
    'ChoiceInteraction',
    'EvolutionInteraction',
    'HyperedgeInteraction',
    'Interaction',
    'NeighborInteraction',
    'PairInteraction',
    'PairPotentialInteraction',
    'PotentialInteraction',
]
