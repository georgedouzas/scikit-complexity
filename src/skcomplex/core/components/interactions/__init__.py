"""Define interactions.

Interactions are the parts of a system whose laws couple elements to each other or to the environment.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import SCHEDULES, BaseInteraction
from ._cascade import CascadeInteraction
from ._connection import ConnectionInteraction
from ._conservation import ConservationInteraction
from ._coupling import CouplingInteraction
from ._crossing import CrossingInteraction
from ._euler_lagrange import EulerLagrangeInteraction
from ._feedback import FeedbackInteraction
from ._geodesic import GeodesicInteraction
from ._gradient import GradientInteraction
from ._grouping import GroupingInteraction
from ._hamiltonian import HamiltonianInteraction
from ._hyperedge import HyperedgeInteraction
from ._interaction import Interaction
from ._jump import JumpInteraction
from ._local_rule import LocalRuleInteraction
from ._map import MapInteraction
from ._neighbor import NeighborInteraction
from ._optimization import OptimizationInteraction
from ._pair import PairInteraction
from ._rate import RateInteraction
from ._sampling import SamplingInteraction
from ._satisfiability import SatisfiabilityInteraction
from ._topology import TopologyInteraction
from ._training import OPTIMIZERS, TrainingInteraction
from ._transfer import TransferInteraction
from ._transition import TransitionInteraction

__all__: list[str] = [
    'OPTIMIZERS',
    'SCHEDULES',
    'BaseInteraction',
    'CascadeInteraction',
    'ConnectionInteraction',
    'ConservationInteraction',
    'CouplingInteraction',
    'CrossingInteraction',
    'EulerLagrangeInteraction',
    'FeedbackInteraction',
    'GeodesicInteraction',
    'GradientInteraction',
    'GroupingInteraction',
    'HamiltonianInteraction',
    'HyperedgeInteraction',
    'Interaction',
    'JumpInteraction',
    'LocalRuleInteraction',
    'MapInteraction',
    'NeighborInteraction',
    'OptimizationInteraction',
    'PairInteraction',
    'RateInteraction',
    'SamplingInteraction',
    'SatisfiabilityInteraction',
    'TopologyInteraction',
    'TrainingInteraction',
    'TransferInteraction',
    'TransitionInteraction',
]
