"""Define the batch matching spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class BatchMatchingSpec(BaseSpec):
    """Batch matching that collects every offer, then matches the largest quantity at one clearing value.

    Requesters are sorted by decreasing and providers by increasing limit, and they are paired while the requester's
    limit covers the provider's. The value is the midpoint of the last paired limits, and every paired requester and
    provider matches at it.

    Examples:
        >>> from skcomplex.specs import BatchMatchingSpec
        >>> BatchMatchingSpec().check()
        BatchMatchingSpec()
    """
