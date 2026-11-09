"""Define the stable matching spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ..types import Preference
from ._base_spec import BaseSpec
from ._random_search_spec import CLEARINGS


@dataclass(frozen=True)
class StableMatchingSpec(BaseSpec):
    """Stable matching of Gale and Shapley.

    Requesters propose to providers, which take as many as they offer units. Requesters rank the providers whose reserve
    they accept by `requester_preference(requester, provider)`, by default the lower reserve, and providers rank the
    requesters by `provider_preference(provider, requester)`, by default the higher reserve. The requester-optimal stable
    matching, from the hospital-resident solver of the `matching` library, pairs every requester with at most one
    provider, and each pair matches one unit at the value of the value rule.

    Args:
        requester_preference:
            The score of a provider for a requester, a function of their agent views, higher for preferred providers,
            or `None` for the lower reserve.

        provider_preference:
            The score of a requester for a provider, a function of their agent views, higher for preferred
            requesters, or `None` for the higher reserve.

        value:
            The value of a match, `'provider'`, `'requester'`, or `'midpoint'`.

    Raises:
        ValidationError:
            The value rule is unknown, raised when its owner checks it.

    Examples:
        >>> from skcomplex.core.specs import StableMatchingSpec
        >>> StableMatchingSpec(value='requester').value
        'requester'
    """

    requester_preference: Preference | None = None
    provider_preference: Preference | None = None
    value: str = 'provider'

    def _check(self) -> None:
        """Raise an error for an unknown value rule.

        Raises:
            ValidationError:
                The value rule is unknown.
        """
        if self.value not in CLEARINGS:
            error_msg = f'Unknown value rule {self.value!r}. Use one of {CLEARINGS}.'
            raise ValidationError(error_msg)
