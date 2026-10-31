"""Define the queue, in which requesters are served by the lowest providers until the quantities run out."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import MutableMapping

import numpy as np

from ...specs import MatchSpec
from ._base_matching_protocol import ORDERS, BaseMatchingProtocol
from ._offer import _Offer


class QueueProtocol(BaseMatchingProtocol):
    """Queue of requesters served by the lowest providers until the quantities run out, so the last are rationed.

    Requesters queue in random order, by decreasing limit, or in order of creation, and each matches at the limit of
    the lowest providers whose limit it accepts.

    Args:
        order:
            The order of the queue, `'random'`, `'limit'`, or `'sequential'`.

    Raises:
        ValidationError:
            The order is unknown, raised by `check`.

    Examples:
        >>> from skcomplex.core.interactions import QueueProtocol
        >>> QueueProtocol(order='limit').check().order
        'limit'
    """

    def __init__(
        self,
        order: str = 'random',
    ) -> None:
        self.order = order

    def _check(self) -> None:
        """Check the order of the queue."""
        self._check_option(self.order, ORDERS, 'order')

    def _match(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches of the queue.

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
        queue = self._order(requesters, self.order, rng)
        stock = sorted(providers, key=lambda provider: provider.limit)
        trades: list[MatchSpec] = []
        for requester in queue:
            for provider in stock:
                if requester.quantity <= 0 or provider.limit > requester.limit:
                    break
                if provider.quantity > 0:
                    self._book(trades, requester, provider, provider.limit)
        return trades
