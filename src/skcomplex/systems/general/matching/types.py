"""Define the types of the matchings.

The type is the rule of a matching, one of the specs of the matching algorithms.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.specs import PartnerSpec
from ._batch_matching_spec import BatchMatchingSpec
from ._continuous_matching_spec import ContinuousMatchingSpec
from ._iterative_matching_spec import IterativeMatchingSpec
from ._partner_switch_spec import PartnerSwitchSpec
from ._queue_spec import QueueSpec
from ._random_search_spec import RandomSearchSpec
from ._stable_matching_spec import StableMatchingSpec

type MatchingRule = (
    RandomSearchSpec
    | PartnerSpec
    | PartnerSwitchSpec
    | QueueSpec
    | IterativeMatchingSpec
    | ContinuousMatchingSpec
    | BatchMatchingSpec
    | StableMatchingSpec
)
