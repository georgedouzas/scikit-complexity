"""Define greedy matching, in which every requester takes the feasible provider of the highest score."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import MutableMapping

import numpy as np

from ...errors import ValidationError
from ...specs import MatchSpec
from ..types import AgentViewProtocol, Preference
from ._base_matching_protocol import CLEARINGS, BaseMatchingProtocol
from ._offer import _Offer
from ._utils import get_location_key

CANDIDATES = ('all', 'neighbors')


class GreedyProtocol(BaseMatchingProtocol):
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
            The candidates or the clearing rule are unknown, or the distance is not positive, raised by `check`.

    Examples:
        >>> from skcomplex.core.interactions import GreedyProtocol
        >>> GreedyProtocol(candidates=2.0, capacity='slots').check().capacity
        'slots'
    """

    def __init__(
        self,
        score: Preference | None = None,
        candidates: str | float = 'all',
        capacity: str | None = None,
        clearing: str = 'provider',
    ) -> None:
        self.score = score
        self.candidates = candidates
        self.capacity = capacity
        self.clearing = clearing

    def _check(self) -> None:
        """Check the candidates, the clearing rule, and the score.

        Raises:
            ValidationError:
                The candidates or the clearing rule are unknown, the distance is not positive, or the score cannot be
                called.
        """
        self._check_option(self.clearing, CLEARINGS, 'clearing')
        if isinstance(self.candidates, str):
            self._check_option(self.candidates, CANDIDATES, 'candidates')
        elif isinstance(self.candidates, bool) or not self.candidates > 0:
            error_msg = f'The candidates are within the distance {self.candidates!r}. Give a positive distance.'
            raise ValidationError(error_msg)
        if self.score is not None and not callable(self.score):
            error_msg = f'The score is {self.score!r}. Give a function of the requester and the provider, or None.'
            raise ValidationError(error_msg)

    def _is_candidate(self, requester: AgentViewProtocol, provider: AgentViewProtocol) -> bool:
        """Check whether a provider is a candidate of a requester."""
        if self.candidates == 'all':
            return True
        if self.candidates == 'neighbors':
            cells = [requester.location, *requester.neighbor_cells()]
            return get_location_key(provider.location) in {get_location_key(cell) for cell in cells}
        return requester.distance_to(provider) <= float(self.candidates)

    def _get_score(self, requester: _Offer, provider: _Offer) -> float:
        """Return the score of a provider for a requester, higher for a preferred provider."""
        if self.score is None:
            return -provider.limit
        return float(self.score(requester.agent, provider.agent))

    def _get_feasible(self, requester: _Offer, providers: list[_Offer], taken: dict[int, int]) -> list[_Offer]:
        """Return the providers a requester can still match with."""
        feasible = []
        for provider in providers:
            room = np.inf if self.capacity is None else float(getattr(provider.agent, self.capacity))
            if provider.quantity <= 0 or taken.get(id(provider), 0) >= room or provider.limit > requester.limit:
                continue
            if self._is_candidate(requester.agent, provider.agent):
                feasible.append(provider)
        return feasible

    def _match(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches of every requester with its best feasible provider.

        Args:
            requesters:
                The offers of the requesters.

            providers:
                The offers of the providers.

            rng:
                The random generator.

            storage:
                The storage of the matching that persists across rounds.

        Returns:
            trades:
                The matches.
        """
        trades: list[MatchSpec] = []
        taken: dict[int, int] = {}
        for requester in self._order(requesters, 'random', rng):
            feasible = self._get_feasible(requester, providers, taken)
            if not feasible:
                continue
            scores = np.asarray([self._get_score(requester, provider) for provider in feasible], dtype=float)
            best = np.flatnonzero(scores == scores.max())
            provider = feasible[int(best[int(rng.integers(len(best)))])]
            matched = len(trades)
            self._book(trades, requester, provider, self._get_value(requester, provider, self.clearing))
            if len(trades) > matched:
                taken[id(provider)] = taken.get(id(provider), 0) + 1
        return trades
