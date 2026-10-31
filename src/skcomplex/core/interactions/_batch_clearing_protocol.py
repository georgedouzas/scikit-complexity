"""Define batch clearing, which collects every offer and matches the largest quantity at one clearing value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import MutableMapping

import numpy as np

from ...specs import MatchSpec
from ._base_matching_protocol import BaseMatchingProtocol
from ._offer import _Offer


class BatchClearingProtocol(BaseMatchingProtocol):
    """Batch clearing that collects every offer, then matches the largest quantity at one clearing value.

    Requesters are sorted by decreasing and providers by increasing limit, and they are paired while the requester's
    limit covers the provider's. The value is the midpoint of the last paired limits, and every paired requester and
    provider matches at it.

    Examples:
        >>> from skcomplex.core.interactions import BatchClearingProtocol
        >>> BatchClearingProtocol().check()
        BatchClearingProtocol()
    """

    @staticmethod
    def _get_requested(offer: _Offer) -> float:
        """Return the quantity a requester can match at its own limit."""
        return offer.quantity if not math.isfinite(offer.budget) else offer.get_affordable(offer.limit)

    def _pair(self, requesters: list[_Offer], providers: list[_Offer]) -> tuple[list[tuple[_Offer, _Offer]], float]:
        """Return the pairs of crossing offers in order of their limits and the value of the last pair."""
        ranked_requesters = sorted(requesters, key=lambda offer: (-offer.limit, offer.agent.id))
        ranked_providers = sorted(providers, key=lambda offer: (offer.limit, offer.agent.id))
        left = {id(offer): self._get_requested(offer) for offer in ranked_requesters}
        offered = {id(offer): offer.quantity for offer in ranked_providers}
        pairs: list[tuple[_Offer, _Offer]] = []
        marginal = math.nan
        requester_iter, provider_iter = iter(ranked_requesters), iter(ranked_providers)
        requester, provider = next(requester_iter, None), next(provider_iter, None)
        while requester is not None and provider is not None and requester.limit >= provider.limit:
            pairs.append((requester, provider))
            marginal = (requester.limit + provider.limit) / 2
            quantity = min(left[id(requester)], offered[id(provider)])
            left[id(requester)] -= quantity
            offered[id(provider)] -= quantity
            if left[id(requester)] <= 0:
                requester = next(requester_iter, None)
            if offered[id(provider)] <= 0:
                provider = next(provider_iter, None)
        return pairs, marginal

    def _match(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches at the uniform value.

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
        pairs, marginal = self._pair(requesters, providers)
        trades: list[MatchSpec] = []
        for requester, provider in pairs:
            self._book(trades, requester, provider, marginal)
        return trades
