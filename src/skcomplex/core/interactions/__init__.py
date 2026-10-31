"""Define interactions, the parts of a system whose laws couple elements to each other or to the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseInteraction
from ._base_matching_protocol import BaseMatchingProtocol
from ._clearing_spec import ClearingSpec
from ._feedback import FeedbackInteraction
from ._hyperedge import HyperedgeInteraction
from ._interaction import Interaction
from ._matching import MatchingInteraction
from ._neighbor import NeighborInteraction
from ._pair import PairInteraction
from ._pair_potential import PairPotentialInteraction
from ._partner_switch_protocol import PartnerSwitchProtocol
from ._potential import PotentialInteraction
from ._price_adjustment_protocol import PriceAdjustmentProtocol
from ._price_time_priority_protocol import PriceTimePriorityProtocol
from ._queue_protocol import QueueProtocol
from ._random_search_protocol import RandomSearchProtocol
from ._stable_matching_protocol import StableMatchingProtocol
from ._uniform_price_protocol import UniformPriceProtocol

__all__: list[str] = [
    'BaseInteraction',
    'BaseMatchingProtocol',
    'ClearingSpec',
    'FeedbackInteraction',
    'HyperedgeInteraction',
    'Interaction',
    'MatchingInteraction',
    'NeighborInteraction',
    'PairInteraction',
    'PairPotentialInteraction',
    'PartnerSwitchProtocol',
    'PotentialInteraction',
    'PriceAdjustmentProtocol',
    'PriceTimePriorityProtocol',
    'QueueProtocol',
    'RandomSearchProtocol',
    'StableMatchingProtocol',
    'UniformPriceProtocol',
]
