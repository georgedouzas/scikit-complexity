"""Define random search, in which requesters meet random providers and match when their limits cross."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import MutableMapping

import numpy as np

from ...specs import MatchSpec
from ._base_matching_protocol import CLEARINGS, BaseMatchingProtocol
from ._offer import _Offer


class RandomSearchProtocol(BaseMatchingProtocol):
    """Random search: requesters in random order meet random providers and match when their limits cross.

    Every requester makes up to `tries` meetings with providers that still offer something, and matches at the
    midpoint of the two limits, or at the requester's or the provider's limit, the protocol of decentralised exchange
    of Hamill and Gilbert (2016 ch. 5).

    Args:
        tries:
            The number of providers every requester meets.

        clearing:
            The value of a match, `'midpoint'`, `'requester'`, or `'provider'`.

    Raises:
        ValidationError:
            The clearing rule is unknown, raised by `check`.

    Examples:
        >>> from skcomplex.core.interactions import RandomSearchProtocol
        >>> RandomSearchProtocol(tries=3).check().tries
        3
    """

    def __init__(
        self,
        tries: int = 1,
        clearing: str = 'midpoint',
    ) -> None:
        self.tries = tries
        self.clearing = clearing

    def _check(self) -> None:
        """Check the clearing rule."""
        self._check_option(self.clearing, CLEARINGS, 'clearing')

    def _match(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches of random meetings.

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
        trades: list[MatchSpec] = []
        for index in rng.permutation(len(requesters)):
            requester = requesters[index]
            for _ in range(self.tries):
                available = [provider for provider in providers if provider.quantity > 0]
                if requester.quantity <= 0 or not available:
                    break
                provider = available[int(rng.integers(len(available)))]
                if provider.limit <= requester.limit:
                    self._book(trades, requester, provider, self._get_value(requester, provider, self.clearing))
        return trades
