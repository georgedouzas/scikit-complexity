"""Define the spec of a continuous matching by limit and arrival."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.errors import ValidationError
from ....core.specs import BaseSpec

ARRIVALS = ('random', 'sequential')


@dataclass(frozen=True)
class ContinuousMatchingSpec(BaseSpec):
    """A spec of a continuous matching, where arriving offers take the best resting.

    Every participant submits one offer, in random order or in order of creation. An arriving offer matches the best
    resting offers of the other side, the oldest first among equal limits, at their limits, as long as the limits cross,
    and its remainder rests. The resting offers are dropped at the end of every matching, as in Caiani et al. (2016,
    chapter 3).

    Unlike `BatchMatchingSpec`, which clears at one value, it matches at many values in order of arrival. Choose it when
    offers meet as they come, as in a limit order book.

    It appears in many fields. Stock exchanges trade continuously through a limit order book, and agent-based models of
    macroeconomics match firms and banks this way.

    Args:
        arrival:
            The order of arrival, `'random'` or `'sequential'`.

    Raises:
        ValidationError:
            The order of arrival is unknown, raised when its owner checks it.

    Examples:
        >>> from skcomplex.systems.general.matching import ContinuousMatchingSpec
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
