"""Define the random search spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.errors import ValidationError
from ....core.specs import BaseSpec

CLEARINGS = ('midpoint', 'requester', 'provider')


@dataclass(frozen=True)
class RandomSearchSpec(BaseSpec):
    """Random search of requesters for providers.

    Requesters in random order meet random providers and match when their limits cross. Every requester makes up to
    `tries` meetings with providers that still offer something, and matches at the midpoint of the two limits, or at the
    requester's or the provider's limit, the decentralised exchange of Hamill and Gilbert (2016 ch. 5).

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
