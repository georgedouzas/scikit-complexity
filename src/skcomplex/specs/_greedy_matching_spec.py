"""Define the greedy matching spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ..types import Preference
from ._base_spec import BaseSpec
from ._random_search_spec import CLEARINGS

CANDIDATES = ('all', 'neighbors')


@dataclass(frozen=True)
class GreedyMatchingSpec(BaseSpec):
    """Greedy matching, in which requesters in random order take the feasible provider of the highest score.

    A provider is feasible for a requester when it still offers a quantity, has taken fewer requesters than its
    capacity, is a candidate of the requester, and its limit is at most the limit of the requester. The candidates are
    every provider, the providers in the cell or node of the requester and in those around it, or the providers within
    a distance. A requester takes the feasible provider of the highest `score(requester, provider)`, a function of their
    agent views that defaults to the lower limit, ties broken at random, and matches the largest quantity both can
    match at the value of the clearing rule.

    Args:
        score:
            The score of a provider for a requester, a function of their agent views, higher for preferred providers,
            or `None` for the lower limit.

        candidates:
            The providers a requester considers, `'all'`, `'neighbors'`, or a distance.

        capacity:
            The name of the provider state with the largest number of requesters it takes, or `None` for no limit
            other than its quantity.

        clearing:
            The value of a match, `'provider'`, `'requester'`, or `'midpoint'`.

    Raises:
        ValidationError:
            The candidates or the clearing rule are unknown, the distance is not positive, or the score cannot be
            called, raised by `check`.

    Examples:
        >>> from skcomplex.specs import GreedyMatchingSpec
        >>> GreedyMatchingSpec(candidates=2.0, capacity='slots').check().capacity
        'slots'
    """

    score: Preference | None = None
    candidates: str | float = 'all'
    capacity: str | None = None
    clearing: str = 'provider'

    def _check(self) -> None:
        """Raise an error for unknown candidates or clearing rule, a distance that is not positive, or a bad score.

        Raises:
            ValidationError:
                The candidates or the clearing rule are unknown, the distance is not positive, or the score cannot be
                called.
        """
        if self.clearing not in CLEARINGS:
            error_msg = f'Unknown clearing {self.clearing!r}. Use one of {CLEARINGS}.'
            raise ValidationError(error_msg)
        if isinstance(self.candidates, str):
            if self.candidates not in CANDIDATES:
                error_msg = f'Unknown candidates {self.candidates!r}. Use one of {CANDIDATES}.'
                raise ValidationError(error_msg)
        elif isinstance(self.candidates, bool) or not self.candidates > 0:
            error_msg = f'The candidates are within the distance {self.candidates!r}. Give a positive distance.'
            raise ValidationError(error_msg)
        if self.score is not None and not callable(self.score):
            error_msg = f'The score is {self.score!r}. Give a function of the requester and the provider, or None.'
            raise ValidationError(error_msg)
