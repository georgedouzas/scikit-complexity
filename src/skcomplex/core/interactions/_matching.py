"""Define matching interactions, which match requesters with providers by a protocol and settle every match."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from ...errors import ValidationError
from ...specs import MatchSpec, RoleSpec
from ..base import BasePart
from ..laws import RuleLaw
from ..laws.utils import transfer
from ..spaces import BaseSpace
from ..types import AgentViewProtocol, ModelViewProtocol
from ._base_matching_protocol import BaseMatchingProtocol
from ._clearing_spec import ClearingSpec
from ._interaction import Interaction
from ._offer import _Offer
from ._random_search_protocol import RandomSearchProtocol

SIGNS = (-1, 1)


class MatchingInteraction[ModelT: ModelViewProtocol](Interaction):
    """Matching of the members of a group of requesters with those of a group of providers by a protocol.

    Requesters state the highest price they accept in their `limit` state, the quantity they want in their `quantity`
    state, and what they can spend in their `budget` state, and providers the lowest price they accept and the quantity
    they offer. A name given as a pair names the state of the requesters first and of the providers second, and a
    missing state counts as no limit, one unit, and no budget. Every match lowers the quantities of both sides and
    moves every state of `transfers`: a negative sign moves `price * quantity` from the requester to the provider, and
    a positive sign moves `quantity` from the provider to the requester, so the totals stay the same.
    `on_match(match, model)` adds any other effect. The matching runs once per round of its phase and reports the
    quantity-weighted `price`, the matched `volume`, and the `unmatched` quantity left, as the reporters
    `<label>.price`, `<label>.volume`, and `<label>.unmatched`.

    Args:
        label:
            The label of the matching.

        requesters:
            The path of the group of requesters.

        providers:
            The path of the group of providers.

        protocol:
            The protocol that matches the members, such as `PriceTimePriorityProtocol()`, or `None` for
            `RandomSearchProtocol()`.

        limit:
            The state with the limit price of both sides, or a pair of the requesters' and the providers' states.

        quantity:
            The state with the quantity of both sides, or a pair of the requesters' and the providers' states.

        budget:
            The state of a requester with what it can spend.

        transfers:
            The states every match moves, each with the sign `-1` for a payment of the requester or `1` for a
            delivery to the requester.

        phase:
            The phase the matching runs in, or `None` for the default phase.

        on_match:
            A function of every match and the model view, run after the transfers, or `None`.

    Raises:
        ValidationError:
            A transfer has a sign other than `-1` or `1`, raised by `check`.

    Examples:
        >>> from skcomplex.core.interactions import MatchingInteraction, PriceTimePriorityProtocol
        >>> matching = MatchingInteraction(
        ...     'exchange',
        ...     requesters='traders',
        ...     providers='traders',
        ...     protocol=PriceTimePriorityProtocol(),
        ...     limit=('bid', 'ask'),
        ... )
        >>> matching.check().reports
        ('price', 'volume', 'unmatched')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'volume', 'unmatched')

    def __init__(
        self,
        label: str,
        *,
        requesters: Annotated[str, RoleSpec()],
        providers: Annotated[str, RoleSpec()],
        protocol: BaseMatchingProtocol | None = None,
        limit: str | tuple[str, str] = 'limit',
        quantity: str | tuple[str, str] = 'quantity',
        budget: str = 'budget',
        transfers: Sequence[tuple[str, int]] = (),
        phase: str | None = None,
        on_match: Callable[[MatchSpec, ModelT], None] | None = None,
    ) -> None:
        super().__init__(label)
        self.requesters = requesters
        self.providers = providers
        self.protocol = protocol
        self.limit = limit
        self.quantity = quantity
        self.budget = budget
        self.transfers = transfers
        self.phase = phase
        self.on_match = on_match

    @staticmethod
    def _get_sides(name: str | tuple[str, str]) -> tuple[str, str]:
        """Return the state of the requesters and of the providers of a name given once or as a pair."""
        return (name, name) if isinstance(name, str) else name

    @staticmethod
    def _read(agent: AgentViewProtocol, name: str, default: float) -> float:
        """Return a state of an agent, or a default when it has none."""
        return float(getattr(agent, name, default))

    def _get_offers(self, model: ModelT) -> tuple[list[_Offer], list[_Offer]]:
        """Return the offers of the requesters and of the providers with a positive quantity."""
        limits, quantities = self._get_sides(self.limit), self._get_sides(self.quantity)
        requesters = [
            _Offer(
                agent,
                self._read(agent, limits[0], math.inf),
                self._read(agent, quantities[0], 1.0),
                self._read(agent, self.budget, math.inf),
            )
            for agent in model.agents('requesters')
        ]
        providers = [
            _Offer(agent, self._read(agent, limits[1], 0.0), self._read(agent, quantities[1], 1.0))
            for agent in model.agents('providers')
        ]
        return [offer for offer in requesters if offer.quantity > 0], [
            offer for offer in providers if offer.quantity > 0
        ]

    def _settle(self, match: MatchSpec, model: ModelT) -> None:
        """Move the transfers of a match, lower the quantities of both sides, and run `on_match`."""
        for state, sign in self.transfers:
            if sign < 0:
                transfer(match.requester, match.provider, match.quantity * match.price, state)
            else:
                transfer(match.provider, match.requester, match.quantity, state)
        for agent, name in zip((match.requester, match.provider), self._get_sides(self.quantity), strict=True):
            if hasattr(agent, name):
                setattr(agent, name, getattr(agent, name) - match.quantity)
        if self.on_match is not None:
            self.on_match(match, model)

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, model: ModelT) -> None:
        """Report the quantity-weighted price, the matched volume, and the unmatched quantity of one clearing.

        Args:
            matches:
                The matches of the clearing.

            unmatched:
                The quantity of both sides left after the clearing.

            model:
                The model view of the matching.
        """
        volume = sum(match.quantity for match in matches)
        price = sum(match.quantity * match.price for match in matches) / volume if volume else math.nan
        model.report('price', price)
        model.report('volume', volume)
        model.report('unmatched', unmatched)

    def _check(self) -> None:
        """Raise an error for a declaration the matching cannot take, a transfer without a sign, or a bad protocol.

        Raises:
            ValidationError:
                A transfer has a sign other than `-1` or `1`, or the protocol has an unknown option.
        """
        super()._check()
        for state, sign in self.transfers:
            if sign not in SIGNS:
                error_msg = f'The transfer of {state!r} has the sign {sign!r}. Use one of {SIGNS}.'
                raise ValidationError(error_msg)
        if self.protocol is not None:
            self.protocol.check()

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Prepare the protocol that matches the members.

        Args:
            context:
                The space the matching inherits from its system, which a matching does not read.
        """
        if self.protocol is not None:
            self.protocol.prepare()

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the rule that matches the members in the matching's phase.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return [RuleLaw(self.match, phase=self.phase)]

    def match(self, model: ModelT) -> None:
        """Match the members, settle the matches, and report the price, the volume, and the unmatched quantity.

        Args:
            model:
                The model view of the matching.
        """
        requesters, providers = self._get_offers(model)
        protocol = RandomSearchProtocol().check().prepare() if self.protocol is None else self.protocol
        matches = protocol.run(ClearingSpec(requesters, providers, model.rng, model.storage)).trades_
        for match in matches:
            self._settle(match, model)
        unmatched = sum(offer.quantity for offer in requesters + providers)
        self._report_matches(matches, unmatched, model)
