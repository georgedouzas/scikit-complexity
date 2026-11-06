"""Define the random search spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

CLEARINGS = ('midpoint', 'requester', 'provider')


@dataclass(frozen=True)
class RandomSearchSpec(BaseSpec):
    """Random search: requesters in random order meet random providers and match when their limits cross.

    Every requester makes up to `tries` meetings with providers that still offer something, and matches at the
    midpoint of the two limits, or at the requester's or the provider's limit, the decentralised exchange of Hamill and
    Gilbert (2016 ch. 5).

    Args:
        tries:
            The number of providers every requester meets.

        clearing:
            The value of a match, `'midpoint'`, `'requester'`, or `'provider'`.

    Raises:
        ValidationError:
            The clearing rule is unknown, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.specs import RandomSearchSpec
        >>> RandomSearchSpec(tries=3).tries
        3
    """

    tries: int = 1
    clearing: str = 'midpoint'

    def _check(self) -> None:
        """Raise an error for an unknown clearing rule.

        Raises:
            ValidationError:
                The clearing rule is unknown.
        """
        if self.clearing not in CLEARINGS:
            error_msg = f'Unknown clearing {self.clearing!r}. Use one of {CLEARINGS}.'
            raise ValidationError(error_msg)
