"""Barter in which every trader trades with the neighbour it matched, the one whose rate differs the most."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Annotated, ClassVar

from sage.all__sagemath_symbolics import NN

from ....core.interactions import MatchingInteraction
from ....core.laws.utils import transfer
from ....core.types import MatchingRule
from ....engine.solvers import ModelView
from ....specs import (
    MatchSpec,
    ParameterSpec,
    RandomSearchSpec,
    RoleSpec,
)
from ._barter_trader import BarterTraderElement
from ._barter_trader_view import BarterTraderView


class BilateralBarterInteraction(MatchingInteraction[ModelView]):
    """Barter in which every trader trades with the neighbour it matched, the one whose rate differs the most.

    A greedy matching pairs every trader, in random order, with the neighbour whose marginal rate of substitution
    differs the most in logarithm, and a neighbour takes any number of partners. The score of a trader that is not a
    neighbour is minus infinity, and a trader without neighbours trades with no one. Each pair trades single units of
    the first good at the price `(g2 + g2') / (g1 + g1')` of their pooled bundles, which lies between their rates, while
    the trade raises the utility of both. Besides the reporters of a matching, the reporters `trades` and `searches`
    count the units traded and the neighbours searched in a step.

    Args:
        label:
            The label of the interaction.

        requesters:
            The path of the group of traders that search for a partner.

        providers:
            The path of the group of traders that can be found, the same group.

        rule:
            The spec of the algorithm that pairs the traders, such as `GreedyMatchingSpec(candidates='neighbors')`.

        limit:
            The state with the limit of both sides, missing for traders, which then accept any partner.

        quantity:
            The states of the partners a requester wants and a provider takes, one and without limit for traders.

        budget:
            The state of a requester with the most value it can give, missing for traders.

        transfers:
            The states every match moves, none for traders, whose trades move the goods.

        phase:
            The phase the barter runs in, or `None` for the default phase.

        on_match:
            A function of every match and the model view, run after the trade, or `None`.

        units:
            The most units one pair trades in a step.

    Examples:
        >>> from skcomplex.models.economics.markets import BilateralBarterInteraction
        >>> BilateralBarterInteraction('barter', requesters='traders', providers='traders').units
        100
    """

    reports: ClassVar[tuple[str, ...]] = (*MatchingInteraction.reports, 'trades', 'searches')

    def __init__(
        self,
        label: str,
        *,
        requesters: Annotated[str, RoleSpec()],
        providers: Annotated[str, RoleSpec()],
        rule: MatchingRule = RandomSearchSpec(),
        limit: str | tuple[str, str] = 'limit',
        quantity: str | tuple[str, str] = 'quantity',
        budget: str = 'budget',
        transfers: Sequence[tuple[str, int]] = (),
        phase: str | None = None,
        on_match: Callable[[MatchSpec, ModelView], None] | None = None,
        units: Annotated[int, ParameterSpec(domain=NN)] = 100,
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
        self.units = units

    @staticmethod
    def _barter(first: BarterTraderView, second: BarterTraderView, units: int) -> int:
        """Trade single units of the first good at the bundle price while both traders gain, and return the units."""
        buyer, seller = (first, second) if first.good2 * second.good1 > second.good2 * first.good1 else (second, first)
        price = (buyer.good2 + seller.good2) / (buyer.good1 + seller.good1)
        traded = 0
        while traded < units:
            buyer_gains = (buyer.good1 + 1) * (buyer.good2 - price) > buyer.good1 * buyer.good2
            seller_gains = (seller.good1 - 1) * (seller.good2 + price) > seller.good1 * seller.good2
            if not (buyer_gains and seller_gains):
                break
            transfer(seller, buyer, 1.0, 'good1')
            transfer(buyer, seller, price, 'good2')
            traded += 1
        return traded

    def _settle(self, match: MatchSpec, model: ModelView) -> None:
        """Trade along a pair of partners, count the units traded, and settle the match."""
        first, second = match.requester, match.provider
        if not isinstance(first, BarterTraderView) or not isinstance(second, BarterTraderView):
            super()._settle(match, model)
            return
        if BarterTraderElement.is_partner(first, second):
            traded = self._barter(first, second, int(model.parameter('units')))
            model.storage['trades'] = self._get_trades(model) + traded
        super()._settle(match, model)

    @staticmethod
    def _get_trades(model: ModelView) -> float:
        """Return the units traded so far in the step, kept in the storage of the barter."""
        trades = model.storage.get('trades')
        return float(trades) if isinstance(trades, int | float) else 0.0

    def match(self, model: ModelView) -> None:
        """Pair and trade, and report the units traded and the neighbours searched besides the matching reporters.

        Args:
            model:
                The model view of the interaction.
        """
        model.storage['trades'] = 0.0
        super().match(model)
        model.report('trades', self._get_trades(model))
        traders = model.agents('requesters', BarterTraderView)
        model.report('searches', float(sum(trader.links for trader in traders)))
