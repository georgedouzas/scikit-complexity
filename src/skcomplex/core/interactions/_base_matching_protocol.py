"""Define the base of matching protocols, which match the offers of requesters with those of providers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod
from collections.abc import MutableMapping

import numpy as np

from ...base import BaseConfig
from ...errors import ValidationError
from ...specs import MatchSpec
from ._matching_round_spec import MatchingRoundSpec
from ._offer import _Offer

CLEARINGS = ('midpoint', 'requester', 'provider')
ORDERS = ('random', 'limit', 'sequential')


class BaseMatchingProtocol(BaseConfig[None, MatchingRoundSpec]):
    """Base class of matching protocols, which match requesters with providers and set the matches.

    A protocol is checked once and run on the offers of every clearing, `protocol.run(MatchingRoundSpec(requesters,
    providers, rng, storage))`, and sets the attribute `trades_`, whose quantities respect the quantities and budgets
    of the offers. A child validates its options in `_check` and matches the offers in `_match`, which is called only
    when both sides have offers.

    Attributes:
        trades_:
            The matches of the last clearing.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.interactions import BaseMatchingProtocol, MatchingRoundSpec
        >>> class NoMatchingProtocol(BaseMatchingProtocol):
        ...     def _match(self, requesters, providers, rng, storage):
        ...         return []
        >>> NoMatchingProtocol().check().prepare().run(MatchingRoundSpec([], [], np.random.default_rng(0), {})).trades_
        []
    """

    trades_: list[MatchSpec]

    @staticmethod
    def _check_option(value: str, allowed: tuple[str, ...], name: str) -> None:
        """Raise an error for an option outside the allowed ones."""
        if value not in allowed:
            error_msg = f'Unknown {name} {value!r}. Use one of {allowed}.'
            raise ValidationError(error_msg)

    @staticmethod
    def _book(trades: list[MatchSpec], requester: _Offer, provider: _Offer, value: float) -> None:
        """Record a match of the largest quantity both sides can match at a value and lower both offers."""
        quantity = min(requester.get_affordable(value), provider.quantity)
        if quantity <= 0:
            return
        requester.quantity -= quantity
        requester.budget -= quantity * value
        provider.quantity -= quantity
        trades.append(MatchSpec(requester.agent, provider.agent, quantity, value))

    @staticmethod
    def _get_value(requester: _Offer, provider: _Offer, clearing: str) -> float:
        """Return the value of a match under a clearing rule."""
        if clearing == 'requester':
            return requester.limit
        if clearing == 'provider':
            return provider.limit
        return (requester.limit + provider.limit) / 2

    @staticmethod
    def _order(offers: list[_Offer], order: str, rng: np.random.Generator) -> list[_Offer]:
        """Return offers in random order, by decreasing limit, or in order of creation."""
        if order == 'limit':
            return sorted(offers, key=lambda item: (-item.limit, item.agent.id))
        if order == 'sequential':
            return sorted(offers, key=lambda item: item.agent.id)
        return [offers[index] for index in rng.permutation(len(offers))]

    def _check(self) -> None:
        """Accept every setting, since the base protocol has none to validate."""

    def _prepare(self, context: None) -> None:
        """Derive nothing from the context, since a protocol prepares nothing."""

    def _run(self, inputs: MatchingRoundSpec | None) -> None:
        """Set the matches of the offers of one clearing, none when a side has no offers.

        Args:
            inputs:
                The offers of the clearing.

        Raises:
            ValidationError:
                No offers were given.
        """
        if inputs is None:
            error_msg = 'A matching protocol runs on the offers of a clearing. Pass a `MatchingRoundSpec`.'
            raise ValidationError(error_msg)
        self.trades_ = []
        if inputs.requesters and inputs.providers:
            self.trades_ = self._match(inputs.requesters, inputs.providers, inputs.rng, inputs.storage)

    @abstractmethod
    def _match(
        self,
        requesters: list[_Offer],
        providers: list[_Offer],
        rng: np.random.Generator,
        storage: MutableMapping[str, object],
    ) -> list[MatchSpec]:
        """Return the matches between requesters and providers that both have offers.

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
