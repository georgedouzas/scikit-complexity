"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._alignment import AlignmentInteraction
from ._base import BaseInteraction
from ._base_matching_protocol import BaseMatchingProtocol
from ._cascade import CascadeInteraction
from ._choice import ChoiceInteraction
from ._clearing_spec import ClearingSpec
from ._constant_force import ConstantForceInteraction
from ._coupling import CouplingInteraction
from ._crossing import CrossingInteraction
from ._damping import DampingInteraction
from ._encounter import EncounterInteraction
from ._evolution import EvolutionInteraction
from ._feedback import FeedbackInteraction
from ._foraging import ForagingInteraction
from ._growth import GrowthInteraction
from ._hyperedge import HyperedgeInteraction
from ._imitation import ImitationInteraction
from ._impulse import ImpulseInteraction
from ._interaction import Interaction
from ._jump import JumpInteraction
from ._local_rule import LocalRuleInteraction
from ._map import MapInteraction
from ._matching import MatchingInteraction
from ._neighbor import NeighborInteraction
from ._ornstein_uhlenbeck import OrnsteinUhlenbeckInteraction
from ._pair import PairInteraction
from ._pair_force import PairForceInteraction
from ._pair_potential import PairPotentialInteraction
from ._partner_switch_protocol import PartnerSwitchProtocol
from ._potential import PotentialInteraction
from ._price_adjustment_protocol import PriceAdjustmentProtocol
from ._price_time_priority_protocol import PriceTimePriorityProtocol
from ._queue_protocol import QueueProtocol
from ._random_search_protocol import RandomSearchProtocol
from ._random_walk import RandomWalkInteraction
from ._reaction import ReactionInteraction
from ._regrowth import RegrowthInteraction
from ._sampling import SamplingInteraction
from ._spreading import SpreadingInteraction
from ._stable_matching_protocol import StableMatchingProtocol
from ._transition import TransitionInteraction
from ._uniform_price_protocol import UniformPriceProtocol

__all__: list[str] = [
    'AlignmentInteraction',
    'BaseInteraction',
    'BaseMatchingProtocol',
    'CascadeInteraction',
    'ChoiceInteraction',
    'ClearingSpec',
    'ConstantForceInteraction',
    'CouplingInteraction',
    'CrossingInteraction',
    'DampingInteraction',
    'EncounterInteraction',
    'EvolutionInteraction',
    'FeedbackInteraction',
    'ForagingInteraction',
    'GrowthInteraction',
    'HyperedgeInteraction',
    'ImitationInteraction',
    'ImpulseInteraction',
    'Interaction',
    'JumpInteraction',
    'LocalRuleInteraction',
    'MapInteraction',
    'MatchingInteraction',
    'NeighborInteraction',
    'OrnsteinUhlenbeckInteraction',
    'PairForceInteraction',
    'PairInteraction',
    'PairPotentialInteraction',
    'PartnerSwitchProtocol',
    'PotentialInteraction',
    'PriceAdjustmentProtocol',
    'PriceTimePriorityProtocol',
    'QueueProtocol',
    'RandomSearchProtocol',
    'RandomWalkInteraction',
    'ReactionInteraction',
    'RegrowthInteraction',
    'SamplingInteraction',
    'SpreadingInteraction',
    'StableMatchingProtocol',
    'TransitionInteraction',
    'UniformPriceProtocol',
]
