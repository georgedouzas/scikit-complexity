"""Define price adjustment, a tâtonnement to the price that balances the requested and provided quantities."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import MutableMapping

import numpy as np

from ...specs import MatchSpec
from ._base_matching_protocol import BaseMatchingProtocol
from ._offer import _Offer


class PriceAdjustmentProtocol(BaseMatchingProtocol):
    """Tâtonnement to the price that balances the requested and provided quantities, then matches at that price.

    Starting from the last clearing price, or the median of the limits, the price moves by
    `p <- p (1 + adjustment (D - S) / (D + S))` until the relative excess of requests falls below `tolerance`, where
    `D` is the quantity requested by requesters whose limit is at least `p` and `S` the quantity of providers whose
    limit is at most `p`. The long side is rationed in random order (Hamill and Gilbert 2016 ch. 5).

    Args:
        adjustment:
            The speed of the price adjustment.

        iterations:
            The largest number of adjustments per clearing.

        tolerance:
            The relative excess of requests at which the adjustment stops.

    Examples:
        >>> from skcomplex.core.interactions import PriceAdjustmentProtocol
        >>> PriceAdjustmentProtocol(adjustment=0.05).check().iterations
        500
    """

    def __init__(
        self,
        adjustment: float = 0.1,
        iterations: int = 500,
        tolerance: float = 1e-3,
    ) -> None:
        self.adjustment = adjustment
        self.iterations = iterations
        self.tolerance = tolerance

    def _adjust(self, requesters: list[_Offer], providers: list[_Offer], start: float) -> float:
        """Return the price the tâtonnement reaches from a start."""
        price = start
        for _ in range(self.iterations):
            requested = sum(requester.get_affordable(price) for requester in requesters if requester.limit >= price)
            provided = sum(provider.quantity for provider in providers if provider.limit <= price)
            total = requested + provided
            if not total or abs(requested - provided) / total < self.tolerance:
                break
            price *= 1 + self.adjustment * (requested - provided) / total
        return price

    def _clear_at(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        price: float,
        rng: np.random.Generator,
    ) -> list[MatchSpec]:
        """Return the matches at one price between the willing requesters and providers in random order."""
        trades: list[MatchSpec] = []
        willing_requesters = self._order([item for item in requesters if item.limit >= price], 'random', rng)
        willing_providers = self._order([item for item in providers if item.limit <= price], 'random', rng)
        pairs = iter(willing_providers)
        provider = next(pairs, None)
        for requester in willing_requesters:
            while provider is not None and requester.get_affordable(price) > 0:
                self._book(trades, requester, provider, price)
                if provider.quantity <= 0:
                    provider = next(pairs, None)
        return trades

    def _match(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches at the price of the tâtonnement.

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
        limits = [offer.limit for offer in requesters + providers if offer.limit > 0]
        if not limits:
            return []
        last = storage.get('price')
        start = last if isinstance(last, float) and math.isfinite(last) and last > 0 else float(np.median(limits))
        price = self._adjust(requesters, providers, start)
        storage['price'] = price
        return self._clear_at(requesters, providers, price, rng)
