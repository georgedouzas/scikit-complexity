"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseInteraction
from ._constant_force import ConstantForceInteraction
from ._damping import DampingInteraction
from ._hyperedge import HyperedgeInteraction
from ._interaction import Interaction
from ._neighbor import NeighborInteraction
from ._ornstein_uhlenbeck import OrnsteinUhlenbeckInteraction
from ._pair import PairInteraction
from ._pair_force import PairForceInteraction
from ._pair_potential import PairPotentialInteraction
from ._potential import PotentialInteraction

__all__: list[str] = [
    'BaseInteraction',
    'ConstantForceInteraction',
    'DampingInteraction',
    'HyperedgeInteraction',
    'Interaction',
    'NeighborInteraction',
    'OrnsteinUhlenbeckInteraction',
    'PairForceInteraction',
    'PairInteraction',
    'PairPotentialInteraction',
    'PotentialInteraction',
]
