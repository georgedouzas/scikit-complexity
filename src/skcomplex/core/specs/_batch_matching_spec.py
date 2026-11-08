"""Define the batch matching spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class BatchMatchingSpec(BaseSpec):
    """Batch matching of the largest quantity at one clearing value.

    It collects every offer, then matches. Requesters are sorted by decreasing and providers by increasing limit, and
    they are paired while the requester's limit covers the provider's. The value is the midpoint of the last paired
    limits, and every paired requester and provider matches at it.

    Examples:
        >>> from skcomplex.core.specs import BatchMatchingSpec
        >>> BatchMatchingSpec()
        BatchMatchingSpec()
    """
