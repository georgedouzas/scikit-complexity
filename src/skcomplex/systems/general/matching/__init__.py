"""Matching of requesters with providers that systems of several fields share.

A matching pairs requesters with providers by a rule spec through the core match law.
"""

from ._batch_matching_spec import BatchMatchingSpec
from ._continuous_matching_spec import ContinuousMatchingSpec
from ._iterative_matching_spec import IterativeMatchingSpec
from ._match_spec import MatchSpec
from ._matching import MatchingInteraction
from ._partner_matching_spec import PartnerMatchingSpec
from ._partner_switch_spec import PartnerSwitchSpec
from ._queue_spec import QueueSpec
from ._random_search_spec import RandomSearchSpec
from ._stable_matching_spec import StableMatchingSpec

__all__: list[str] = [
    'BatchMatchingSpec',
    'ContinuousMatchingSpec',
    'IterativeMatchingSpec',
    'MatchSpec',
    'MatchingInteraction',
    'PartnerMatchingSpec',
    'PartnerSwitchSpec',
    'QueueSpec',
    'RandomSearchSpec',
    'StableMatchingSpec',
]
