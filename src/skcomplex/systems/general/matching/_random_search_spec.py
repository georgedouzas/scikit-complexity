"""Define the spec of a random search of requesters for providers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.errors import ValidationError
from ....core.specs import BaseSpec

CLEARINGS = ('midpoint', 'requester', 'provider')


@dataclass(frozen=True)
class RandomSearchSpec(BaseSpec):
    """A spec of a random search, where requesters meet random providers and match.

    Every requester, in random order, makes up to `tries` meetings with providers that still offer something, and
    matches when the limits cross, at the midpoint of the two limits or at the limit of one side. It is the
    decentralised exchange of Hamill and Gilbert (2016, chapter 5).

    Unlike `BatchMatchingSpec`, which sees every offer, it lets each requester see only a few. Choose it when
    information is local and search is costly.

    It appears in many fields. Buyers search among sellers in a bazaar, job seekers meet a few firms, and animals search
    for mates.

    Args:
        tries:
            The number of providers every requester meets.

        value:
            The value of a match, `'midpoint'`, `'requester'`, or `'provider'`.

    Raises:
        ValidationError:
            The value rule is unknown, raised when its owner checks it.

    Examples:
        >>> from skcomplex.systems.general.matching import RandomSearchSpec
        >>> RandomSearchSpec(tries=3).tries
        3
    """

    tries: int = 1
    value: str = 'midpoint'

    def _check(self) -> None:
        """Raise an error for an unknown value rule.

        Raises:
            ValidationError:
                The value rule is unknown.
        """
        if self.value not in CLEARINGS:
            error_msg = f'Unknown value rule {self.value!r}. Use one of {CLEARINGS}.'
            raise ValidationError(error_msg)
