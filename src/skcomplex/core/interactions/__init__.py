"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._alignment import AlignmentInteraction
from ._base import BaseInteraction
from ._base_matching_protocol import BaseMatchingProtocol
from ._batch_clearing_protocol import BatchClearingProtocol
from ._choice import ChoiceInteraction
from ._constant_force import ConstantForceInteraction
from ._continuous_clearing_protocol import ContinuousClearingProtocol
from ._coupling import CouplingInteraction
from ._damping import DampingInteraction
from ._evolution import EvolutionInteraction
from ._feedback import FeedbackInteraction
from ._greedy_protocol import GreedyProtocol
from ._grouping import GroupingInteraction
from ._growth import GrowthInteraction
from ._hyperedge import HyperedgeInteraction
from ._imitation import ImitationInteraction
from ._interaction import Interaction
from ._iterative_clearing_protocol import IterativeClearingProtocol
from ._life_cycle import LifeCycleInteraction
from ._matching import MatchingInteraction
from ._matching_round_spec import MatchingRoundSpec
from ._movement import MovementInteraction
from ._neighbor import NeighborInteraction
from ._ornstein_uhlenbeck import OrnsteinUhlenbeckInteraction
from ._pair import PairInteraction
from ._pair_force import PairForceInteraction
from ._pair_potential import PairPotentialInteraction
from ._partner_switch_protocol import PartnerSwitchProtocol
from ._potential import PotentialInteraction
from ._queue_protocol import QueueProtocol
from ._random_search_protocol import RandomSearchProtocol
from ._reaction import ReactionInteraction
from ._spreading import SpreadingInteraction
from ._stable_matching_protocol import StableMatchingProtocol
from ._threshold import ThresholdInteraction
from ._topology import TopologyInteraction
from ._transfer import TransferInteraction

__all__: list[str] = [
    'AlignmentInteraction',
    'BaseInteraction',
    'BaseMatchingProtocol',
    'BatchClearingProtocol',
    'ChoiceInteraction',
    'ConstantForceInteraction',
    'ContinuousClearingProtocol',
    'CouplingInteraction',
    'DampingInteraction',
    'EvolutionInteraction',
    'FeedbackInteraction',
    'GreedyProtocol',
    'GroupingInteraction',
    'GrowthInteraction',
    'HyperedgeInteraction',
    'ImitationInteraction',
    'Interaction',
    'IterativeClearingProtocol',
    'LifeCycleInteraction',
    'MatchingInteraction',
    'MatchingRoundSpec',
    'MovementInteraction',
    'NeighborInteraction',
    'OrnsteinUhlenbeckInteraction',
    'PairForceInteraction',
    'PairInteraction',
    'PairPotentialInteraction',
    'PartnerSwitchProtocol',
    'PotentialInteraction',
    'QueueProtocol',
    'RandomSearchProtocol',
    'ReactionInteraction',
    'SpreadingInteraction',
    'StableMatchingProtocol',
    'ThresholdInteraction',
    'TopologyInteraction',
    'TransferInteraction',
]
