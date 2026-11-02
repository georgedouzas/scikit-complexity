"""Limit order book that matches the orders of the traders and reports the last traded price."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Annotated, ClassVar

from ....core.interactions import MatchingInteraction
from ....core.types import MatchingRule
from ....engine.solvers import ModelView
from ....specs import (
    MatchSpec,
    RandomSearchSpec,
    RoleSpec,
)


class BookInteraction(MatchingInteraction[ModelView]):
    """Limit order book that matches the orders of the traders and reports the last traded price.

    The book is a matching of the traders with themselves, which binds the requesters' limit to their `bid` and the
    providers' to their `ask`, and every trade is two transfers, money from buyer to seller and shares from seller to
    buyer, so the total money and the total shares stay the same, which the default run checks as invariants. The book
    reports the last traded `price`, the previous price when nothing trades, and the traded `volume`.

    Args:
        label:
            The label of the book.

        requesters:
            The path of the group of buying traders.

        providers:
            The path of the group of selling traders.

        rule:
            The spec of the algorithm that matches the orders, such as `ContinuousMatchingSpec()`.

        limit:
            The states with the limit prices of the buyers and of the sellers.

        quantity:
            The states with the quantities the buyers want and the sellers offer.

        budget:
            The state of a buyer with the money it can spend.

        transfers:
            The states every trade moves, money paid by the buyer and shares delivered to it.

        phase:
            The phase the book clears in.

        on_match:
            A function of every trade and the model view, run after the transfers, or `None`.

        opening:
            The price before the first trade.

    Examples:
        >>> from skcomplex.models.economics.finance import BookInteraction
        >>> BookInteraction('book', requesters='traders', providers='traders').reports
        ('price', 'volume')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'volume')

    def __init__(
        self,
        label: str,
        *,
        requesters: Annotated[str, RoleSpec()],
        providers: Annotated[str, RoleSpec()],
        rule: MatchingRule = RandomSearchSpec(),
        limit: str | tuple[str, str] = ('bid', 'ask'),
        quantity: str | tuple[str, str] = ('demand', 'supply'),
        budget: str = 'money',
        transfers: Sequence[tuple[str, int]] = (('money', -1), ('shares', 1)),
        phase: str | None = 'trade',
        on_match: Callable[[MatchSpec, ModelView], None] | None = None,
        opening: float = 100.0,
    ) -> None:
        super().__init__(
            label,
            requesters=requesters,
            providers=providers,
            rule=rule,
            limit=limit,
            quantity=quantity,
            budget=budget,
            transfers=transfers,
            phase=phase,
            on_match=on_match,
        )
        self.opening = opening

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, model: ModelView) -> None:
        """Report the last traded price, the previous price when nothing trades, and the traded volume.

        Args:
            matches:
                The trades of the clearing.

            unmatched:
                The quantity of both sides left after the clearing, which the book does not report.

            model:
                The model view of the book.
        """
        del unmatched
        last = model.history('price', 1)
        price = matches[-1].price if matches else (float(last[-1]) if len(last) else self.opening)
        model.report('price', price)
        model.report('volume', sum(match.quantity for match in matches))
