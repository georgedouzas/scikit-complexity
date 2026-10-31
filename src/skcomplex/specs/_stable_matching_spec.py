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
    """Stable matching of Gale and Shapley, requesters proposing to providers that take as many as they offer units.

    Requesters rank the providers whose limit they accept by `requester_preference(requester, provider)`, by default
    the lower limit, and providers rank the requesters by `provider_preference(provider, requester)`, by default the
    higher limit. The requester-optimal stable matching, from the hospital-resident solver of the `matching` library,
    pairs every requester with at most one provider, and each pair matches one unit at the value of the clearing rule.

    Args:
        requester_preference:
            The score of a provider for a requester, a function of their agent views, higher for preferred providers,
            or `None` for the lower limit.

        provider_preference:
            The score of a requester for a provider, a function of their agent views, higher for preferred
            requesters, or `None` for the higher limit.

        clearing:
            The value of a match, `'provider'`, `'requester'`, or `'midpoint'`.

    Raises:
        ValidationError:
            The clearing rule is unknown, raised by `check`.

    Examples:
        >>> from skcomplex.specs import StableMatchingSpec
        >>> StableMatchingSpec(clearing='requester').check().clearing
        'requester'
    """

    requester_preference: Preference | None = None
    provider_preference: Preference | None = None
    clearing: str = 'provider'

    def _check(self) -> None:
        """Raise an error for an unknown clearing rule.

        Raises:
            ValidationError:
                The clearing rule is unknown.
        """
        if self.clearing not in CLEARINGS:
            error_msg = f'Unknown clearing {self.clearing!r}. Use one of {CLEARINGS}.'
            raise ValidationError(error_msg)
