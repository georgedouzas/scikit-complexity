"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._alignment import AlignmentInteraction
from ._base import BaseInteraction
from ._base_matching_protocol import BaseMatchingProtocol
from ._batch_clearing_protocol import BatchClearingProtocol
from ._cascade import CascadeInteraction
from ._choice import ChoiceInteraction
from ._conservation import ConservationInteraction
from ._continuous_clearing_protocol import ContinuousClearingProtocol
from ._coupling import CouplingInteraction
from ._crossing import CrossingInteraction
from ._diffusion import DiffusionInteraction
from ._encounter import EncounterInteraction
from ._euler_lagrange import EulerLagrangeInteraction
from ._evolution import EvolutionInteraction
from ._feedback import FeedbackInteraction
from ._gradient import GradientInteraction
from ._greedy_protocol import GreedyProtocol
from ._grouping import GroupingInteraction
from ._hyperedge import HyperedgeInteraction
from ._imitation import ImitationInteraction
from ._impulse import ImpulseInteraction
from ._interaction import Interaction
from ._iterative_clearing_protocol import IterativeClearingProtocol
from ._jump import JumpInteraction
from ._life_cycle import LifeCycleInteraction
from ._local_rule import LocalRuleInteraction
from ._map import MapInteraction
from ._matching import MatchingInteraction
from ._matching_round_spec import MatchingRoundSpec
from ._movement import MovementInteraction
from ._neighbor import NeighborInteraction
from ._pair import PairInteraction
from ._pair_force import PairForceInteraction
from ._pair_potential import PairPotentialInteraction
from ._partner_switch_protocol import PartnerSwitchProtocol
from ._queue_protocol import QueueProtocol
from ._random_search_protocol import RandomSearchProtocol
from ._rate import RateInteraction
from ._reaction import ReactionInteraction
from ._reinforcement import ReinforcementInteraction
from ._relaxation import RelaxationInteraction
from ._sampling import SamplingInteraction
from ._spreading import SpreadingInteraction
from ._stable_matching_protocol import StableMatchingProtocol
from ._topology import TopologyInteraction
from ._transfer import TransferInteraction
from ._transition import TransitionInteraction

__all__: list[str] = [
    'AlignmentInteraction',
    'BaseInteraction',
    'BaseMatchingProtocol',
    'BatchClearingProtocol',
    'CascadeInteraction',
    'ChoiceInteraction',
    'ConservationInteraction',
    'ContinuousClearingProtocol',
    'CouplingInteraction',
    'CrossingInteraction',
    'DiffusionInteraction',
    'EncounterInteraction',
    'EulerLagrangeInteraction',
    'EvolutionInteraction',
    'FeedbackInteraction',
    'GradientInteraction',
    'GreedyProtocol',
    'GroupingInteraction',
    'HyperedgeInteraction',
    'ImitationInteraction',
    'ImpulseInteraction',
    'Interaction',
    'IterativeClearingProtocol',
    'JumpInteraction',
    'LifeCycleInteraction',
    'LocalRuleInteraction',
    'MapInteraction',
    'MatchingInteraction',
    'MatchingRoundSpec',
    'MovementInteraction',
    'NeighborInteraction',
    'PairForceInteraction',
    'PairInteraction',
    'PairPotentialInteraction',
    'PartnerSwitchProtocol',
    'QueueProtocol',
    'RandomSearchProtocol',
    'RateInteraction',
    'ReactionInteraction',
    'ReinforcementInteraction',
    'RelaxationInteraction',
    'SamplingInteraction',
    'SpreadingInteraction',
    'StableMatchingProtocol',
    'TopologyInteraction',
    'TransferInteraction',
    'TransitionInteraction',
]
