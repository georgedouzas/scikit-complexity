"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._advection import AdvectionInteraction
from ._base import BaseInteraction
from ._cascade import CascadeInteraction
from ._choice import ChoiceInteraction
from ._conservation import ConservationInteraction
from ._coupling import CouplingInteraction
from ._crossing import CrossingInteraction
from ._diffusion import DiffusionInteraction
from ._encounter import EncounterInteraction
from ._euler_lagrange import EulerLagrangeInteraction
from ._evolution import EvolutionInteraction
from ._feedback import FeedbackInteraction
from ._gradient import GradientInteraction
from ._grouping import GroupingInteraction
from ._hamiltonian import HamiltonianInteraction
from ._hyperedge import HyperedgeInteraction
from ._imitation import ImitationInteraction
from ._impulse import ImpulseInteraction
from ._interaction import Interaction
from ._jump import JumpInteraction
from ._life_cycle import LifeCycleInteraction
from ._local_rule import LocalRuleInteraction
from ._map import MapInteraction
from ._matching import MatchingInteraction
from ._movement import MovementInteraction
from ._neighbor import NeighborInteraction
from ._optimization import OptimizationInteraction
from ._pair import PairInteraction
from ._pair_coupling import PairCouplingInteraction
from ._rate import RateInteraction
from ._reaction import ReactionInteraction
from ._reinforcement import ReinforcementInteraction
from ._relaxation import RelaxationInteraction
from ._sampling import SamplingInteraction
from ._satisfiability import SatisfiabilityInteraction
from ._sign_adjustment import SignAdjustmentInteraction
from ._topology import TopologyInteraction
from ._training import OPTIMIZERS, TrainingInteraction
from ._transfer import TransferInteraction
from ._transition import TransitionInteraction

__all__: list[str] = [
    'OPTIMIZERS',
    'AdvectionInteraction',
    'BaseInteraction',
    'CascadeInteraction',
    'ChoiceInteraction',
    'ConservationInteraction',
    'CouplingInteraction',
    'CrossingInteraction',
    'DiffusionInteraction',
    'EncounterInteraction',
    'EulerLagrangeInteraction',
    'EvolutionInteraction',
    'FeedbackInteraction',
    'GradientInteraction',
    'GroupingInteraction',
    'HamiltonianInteraction',
    'HyperedgeInteraction',
    'ImitationInteraction',
    'ImpulseInteraction',
    'Interaction',
    'JumpInteraction',
    'LifeCycleInteraction',
    'LocalRuleInteraction',
    'MapInteraction',
    'MatchingInteraction',
    'MovementInteraction',
    'NeighborInteraction',
    'OptimizationInteraction',
    'PairCouplingInteraction',
    'PairInteraction',
    'RateInteraction',
    'ReactionInteraction',
    'ReinforcementInteraction',
    'RelaxationInteraction',
    'SamplingInteraction',
    'SatisfiabilityInteraction',
    'SignAdjustmentInteraction',
    'TopologyInteraction',
    'TrainingInteraction',
    'TransferInteraction',
    'TransitionInteraction',
]
