"""Define the continuous matching spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

ARRIVALS = ('random', 'sequential')


@dataclass(frozen=True)
class ContinuousMatchingSpec(BaseSpec):
    """Continuous matching by limit and arrival.

    Every arriving offer matches the best resting offers. Every participant submits one offer, in random order of
    arrival or in order of creation. An arriving offer matches the best offers of the other side, the oldest first among
    equal limits, at their limits, as long as the limits cross, and its remainder rests. The resting offers are dropped
    at the end of every matching (Caiani et al. 2016 ch. 3).

    Args:
        arrival:
            The order of arrival, `'random'` or `'sequential'`.

    Raises:
        ValidationError:
            The order of arrival is unknown, raised when its owner checks it.

    Examples:
        >>> from skcomplex.core.specs import ContinuousMatchingSpec
        >>> ContinuousMatchingSpec().arrival
        'random'
    """

    arrival: str = 'random'

    def _check(self) -> None:
        """Raise an error for an unknown order of arrival.

        Raises:
            ValidationError:
                The order of arrival is unknown.
        """
        if self.arrival not in ARRIVALS:
            error_msg = f'Unknown arrival {self.arrival!r}. Use one of {ARRIVALS}.'
            raise ValidationError(error_msg)
