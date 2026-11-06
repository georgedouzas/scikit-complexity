"""Define the partner spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ..types import Preference
from ._base_spec import BaseSpec
from ._random_search_spec import CLEARINGS

CHOICES = ('uniform', 'score', 'nearest', 'all')
FALLBACKS = (None, 'nearest')


@dataclass(frozen=True)
class PartnerSpec(BaseSpec):
    """Selection of partners: the candidates of every agent, a choice among them, a fallback, and a capacity.

    Every agent of one side considers its candidates among the agents of the other side and picks among those that
    still have room. With a capacity the agents act in random order, since the first to act take the room. Without one
    they act in their order. A capacity of 1 with every agent as a candidate and a uniform choice draws disjoint random
    pairs from one random order per side.

    Args:
        candidates:
            The agents an agent considers, `'all'`, `'neighbors'` for those in the cells or nodes around it, `'cell'`
            for those in its cell or node, `'local'` for both, a distance, or the name of a state of the agent that
            holds the distance.

        choice:
            How an agent picks among its candidates, `'uniform'`, `'score'` for the largest `score` with ties broken
            at random, `'nearest'`, or `'all'` for every candidate.

        fallback:
            The partner of an agent without candidates, `'nearest'` for the nearest agent with room, or `None` for no
            partner.

        capacity:
            The largest number of pairs of an agent in one step, a number, the name of a state of the agent that holds
            it, or `None` for no limit.

        score:
            The function `score(agent, candidate)` of their agent views the choice `'score'` reads, higher for preferred
            candidates, or `None`.

        clearing:
            The value of a pair that a matching settles, `'midpoint'` of both limits, `'provider'`, or `'requester'`.

    Raises:
        ValidationError:
            The choice, the fallback, or the clearing is unknown, the distance or the capacity is not positive, or the
            score does not fit the choice, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.specs import PartnerSpec
        >>> PartnerSpec(candidates='reach', fallback='nearest').candidates
        'reach'
    """

    candidates: str | float = 'all'
    choice: str = 'uniform'
    fallback: str | None = None
    capacity: str | float | None = None
    score: Preference | None = None
    clearing: str = 'midpoint'

    def _check(self) -> None:
        """Raise an error for an unknown choice, fallback, or clearing, a bad distance or capacity, or a bad score.

        Raises:
            ValidationError:
                The choice, the fallback, or the clearing is unknown, the distance or the capacity is not positive, or
                the score does not fit the choice.
        """
        if self.clearing not in CLEARINGS:
            error_msg = f'Unknown clearing {self.clearing!r}. Use one of {CLEARINGS}.'
            raise ValidationError(error_msg)
        if self.choice not in CHOICES:
            error_msg = f'Unknown choice {self.choice!r} of partners. Use one of {CHOICES}.'
            raise ValidationError(error_msg)
        if self.fallback not in FALLBACKS:
            error_msg = f'Unknown fallback {self.fallback!r} of partners. Use one of {FALLBACKS}.'
            raise ValidationError(error_msg)
        if not isinstance(self.candidates, str) and (isinstance(self.candidates, bool) or not self.candidates > 0):
            error_msg = f'The candidates are within the distance {self.candidates!r}. Give a positive distance.'
            raise ValidationError(error_msg)
        capacity = self.capacity
        if capacity is not None and not isinstance(capacity, str) and (isinstance(capacity, bool) or not capacity >= 1):
            error_msg = f'The capacity of partners is {capacity!r}. Give a number of at least 1, a state, or None.'
            raise ValidationError(error_msg)
        if (self.choice == 'score') != callable(self.score):
            error_msg = (
                f'The choice {self.choice!r} of partners has the score {self.score!r}. Give the choice `score` a '
                'function of the agent and the candidate, and any other choice no score.'
            )
            raise ValidationError(error_msg)
