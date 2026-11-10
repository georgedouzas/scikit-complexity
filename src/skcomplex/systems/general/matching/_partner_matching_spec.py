"""Define the spec of a choice of partners that a matching settles at a value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.errors import ValidationError
from ....core.specs import PartnerSpec
from ._random_search_spec import CLEARINGS


@dataclass(frozen=True)
class PartnerMatchingSpec(PartnerSpec):
    """A spec of a choice of partners that a matching settles at a value.

    It is a `PartnerSpec` with the value of a pair, which only a matching reads.

    Unlike `PartnerSpec`, which only pairs members, it also says the value at which each pair settles. Choose it when
    members pick partners by distance or score and then trade.

    It appears in many fields. Buyers visit the nearest shops, workers apply to the best paying firms within reach, and
    traders deal with partners they trust.

    Args:
        candidates:
            The agents an agent considers, as in `PartnerSpec`.

        choice:
            How an agent picks among its candidates, as in `PartnerSpec`.

        fallback:
            The partner of an agent without candidates, as in `PartnerSpec`.

        capacity:
            The largest number of pairs of an agent in one step, as in `PartnerSpec`.

        score:
            The function `score(agent, candidate)` the choice `'score'` reads, as in `PartnerSpec`.

        value:
            The value of a pair that a matching settles, `'midpoint'` of both reserves, `'provider'`, or `'requester'`.

    Raises:
        ValidationError:
            The value rule is unknown, or a setting of the partners is invalid, raised when its owner checks it.

    Examples:
        >>> from skcomplex.systems.general.matching import PartnerMatchingSpec
        >>> PartnerMatchingSpec('local', value='provider').value
        'provider'
    """

    value: str = 'midpoint'

    def _check(self) -> None:
        """Raise an error for an unknown value rule or an invalid setting of the partners.

        Raises:
            ValidationError:
                The value rule is unknown, or a setting of the partners is invalid.
        """
        if self.value not in CLEARINGS:
            error_msg = f'Unknown value rule {self.value!r}. Use one of {CLEARINGS}.'
            raise ValidationError(error_msg)
        super()._check()
