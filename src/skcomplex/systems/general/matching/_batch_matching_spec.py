"""Define the spec of a batch matching at one clearing value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.specs import BaseSpec


@dataclass(frozen=True)
class BatchMatchingSpec(BaseSpec):
    """A spec of a batch matching of the largest quantity at one clearing value.

    It collects every offer, then matches. Requesters are sorted by decreasing and providers by increasing limit, and
    they pair while the limit of the requester covers that of the provider. The value is the midpoint of the last paired
    limits, and every pair matches at it.

    Unlike `ContinuousMatchingSpec`, which matches offers as they arrive, it clears all offers at once at one value.
    Choose it when a market clears in rounds, as a call auction does.

    It appears in many fields. Stock exchanges open with a call auction, electricity markets clear once per hour, and
    markets for emission permits clear in rounds.

    Examples:
        >>> from skcomplex.systems.general.matching import BatchMatchingSpec
        >>> BatchMatchingSpec()
        BatchMatchingSpec()
    """
