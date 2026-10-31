"""Define continuous clearing, which matches every arriving offer with the best resting offers of the other side."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import heapq
from collections.abc import MutableMapping

import numpy as np

from ...specs import MatchSpec
from ._base_matching_protocol import BaseMatchingProtocol
from ._offer import _Offer

ARRIVALS = ('random', 'sequential')


class ContinuousClearingProtocol(BaseMatchingProtocol):
    """Continuous clearing by limit and arrival, in which every arriving offer matches the best resting offers.

    Every participant submits one offer, in random order of arrival or in order of creation. An arriving offer matches
    the best offers of the other side, the oldest first among equal limits, at their limits, as long as the limits
    cross, and its remainder rests. The resting offers are dropped at the end of every clearing (Caiani et al. 2016
    ch. 3).

    Args:
        arrival:
            The order of arrival, `'random'` or `'sequential'`.

    Raises:
        ValidationError:
            The order of arrival is unknown, raised by `check`.

    Examples:
        >>> from skcomplex.core.interactions import ContinuousClearingProtocol
        >>> ContinuousClearingProtocol().check().arrival
        'random'
    """

    def __init__(
        self,
        arrival: str = 'random',
    ) -> None:
        self.arrival = arrival

    def _check(self) -> None:
        """Check the order of arrival."""
        self._check_option(self.arrival, ARRIVALS, 'arrival')

    def _arrange(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
    ) -> list[tuple[_Offer, bool]]:
        """Return the offers in their order of arrival, each with whether it is a request."""
        offers = [(offer, True) for offer in requesters] + [(offer, False) for offer in providers]
        if self.arrival == 'sequential':
            return sorted(offers, key=lambda item: item[0].agent.id)
        return [offers[index] for index in rng.permutation(len(offers))]

    def _match(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches of the continuous matching.

        Args:
            requesters:
                The offers of the requesters.

            providers:
                The offers of the providers.

            rng:
                The random generator.

            storage:
                The storage of the matching that persists across clearings.

        Returns:
            trades:
                The matches.
        """
        requests: list[tuple[float, int, _Offer]] = []
        provisions: list[tuple[float, int, _Offer]] = []
        trades: list[MatchSpec] = []
        for time, (offer, is_request) in enumerate(self._arrange(requesters, providers, rng)):
            if is_request:
                while provisions and provisions[0][0] <= offer.limit and offer.get_affordable(provisions[0][0]) > 0:
                    resting = provisions[0][2]
                    self._book(trades, offer, resting, resting.limit)
                    if resting.quantity <= 0:
                        heapq.heappop(provisions)
                if offer.get_affordable(offer.limit) > 0:
                    heapq.heappush(requests, (-offer.limit, time, offer))
                continue
            while requests and -requests[0][0] >= offer.limit and offer.quantity > 0:
                resting = requests[0][2]
                self._book(trades, resting, offer, resting.limit)
                if resting.get_affordable(resting.limit) <= 0:
                    heapq.heappop(requests)
            if offer.quantity > 0:
                heapq.heappush(provisions, (offer.limit, time, offer))
        return trades
