"""Define interactions, the components that change the states of elements.

An interaction reads the states of the elements its roles bind and returns laws that change them, such as a rate, a map,
a random event, or a transfer between members.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseInteraction
from ._connection import ConnectionInteraction
from ._conservation import ConservationInteraction
from ._coupling import CouplingInteraction
from ._crossing import CrossingInteraction
from ._grouping import GroupingInteraction
from ._hyperedge import HyperedgeInteraction
from ._interaction import Interaction
from ._jump import JumpInteraction
from ._local_rule import LocalRuleInteraction
from ._map import MapInteraction
from ._neighbor import NeighborInteraction
from ._pair import PairInteraction
from ._rate import RateInteraction
from ._sampling import SamplingInteraction
from ._topology import TopologyInteraction
from ._training import TrainingInteraction
from ._transfer import TransferInteraction
from ._transition import TransitionInteraction
from ._wiener_process_spec import WienerProcessSpec

__all__: list[str] = [
    'BaseInteraction',
    'ConnectionInteraction',
    'ConservationInteraction',
    'CouplingInteraction',
    'CrossingInteraction',
    'GroupingInteraction',
    'HyperedgeInteraction',
    'Interaction',
    'JumpInteraction',
    'LocalRuleInteraction',
    'MapInteraction',
    'NeighborInteraction',
    'PairInteraction',
    'RateInteraction',
    'SamplingInteraction',
    'TopologyInteraction',
    'TrainingInteraction',
    'TransferInteraction',
    'TransitionInteraction',
    'WienerProcessSpec',
]
