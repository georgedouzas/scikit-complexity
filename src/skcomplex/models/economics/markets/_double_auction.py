"""Markets models: double auctions of zero-intelligence traders with and without budget constraints."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import FeedbackInteraction, MapInteraction, MatchingInteraction
from ....core.system import System
from ....core.times import DiscreteTime
from ....engine.solvers import ModelView
from ....specs import (
    BatchClearingSpec,
    ContinuousClearingSpec,
    MatchSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    StateSpec,
    VisualSpec,
)
from ..._catalog import register_model
from ._zero_intelligence_buyer_view import ZeroIntelligenceBuyerView
from ._zero_intelligence_seller_view import ZeroIntelligenceSellerView

PROTOCOLS = ('order_book', 'call_auction')


def maximum_surplus(values: np.ndarray, costs: np.ndarray) -> float:
    """Return the largest total surplus of single units, the competitive surplus of the supply and demand schedules.

    Args:
        values:
            The value of a unit to every buyer.

        costs:
            The cost of a unit to every seller.

    Returns:
        surplus:
            The sum of the positive differences between the highest values and the lowest costs.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.economics.markets import maximum_surplus
        >>> maximum_surplus(np.array([10.0, 6.0, 2.0]), np.array([1.0, 4.0, 8.0]))
        11.0
    """
    demand, supply = np.sort(values)[::-1], np.sort(costs)
    size = min(len(demand), len(supply))
    return float(np.maximum(demand[:size] - supply[:size], 0).sum())


def _book_profit(trade: MatchSpec, model: ModelView) -> None:
    """Add the surplus of a trade to the profits of the buyer and the seller."""
    del model
    buyer, seller = trade.requester, trade.provider
    if not isinstance(buyer, ZeroIntelligenceBuyerView) or not isinstance(seller, ZeroIntelligenceSellerView):
        return
    buyer.profit += trade.quantity * (buyer.value - trade.price)
    seller.profit += trade.quantity * (trade.price - seller.value)


class ZeroIntelligenceBuyerElement(Element):
    """Buyer of one unit per trading period who bids a random price, below its value under the budget constraint.

    In every round of a trading period, a buyer that has not bought yet bids a price drawn uniformly from 0 to its
    value, or to the price ceiling without the constraint, the zero-intelligence traders of Gode and Sunder (1993).

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The value of the unit to the buyer, its redemption value.

        bid:
            The current bid.

        demand:
            The number of units still wanted in the trading period.

        profit:
            The accumulated surplus of the buyer's trades.

        ceiling:
            The highest price of the market.

        constrained:
            One when the buyer never bids above its value, zero otherwise.

    Examples:
        >>> from skcomplex.models.economics.markets import ZeroIntelligenceBuyerElement
        >>> GroupElement('buyers', element=ZeroIntelligenceBuyerElement, size=5, values={'value': 100.0}).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        bid: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        profit: Annotated[float, StateSpec()] = 0.0,
        ceiling: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 200.0,
        constrained: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
        self.bid = bid
        self.demand = demand
        self.profit = profit
        self.ceiling = ceiling
        self.constrained = constrained

    @classmethod
    def _get_view(cls) -> type[ZeroIntelligenceBuyerView]:
        """Return the view of a buyer, with its value, bid, demand, and profit."""
        return ZeroIntelligenceBuyerView

    @staticmethod
    def get_bid(agent: ZeroIntelligenceBuyerView, model: ModelView) -> float:
        """Return a bid drawn uniformly from 0 to the value, or to the price ceiling without the constraint.

        Args:
            agent:
                The buyer.

            model:
                The model view of the quotes.

        Returns:
            bid:
                The bid of the round.
        """
        return float(model.rng.uniform(0.0, agent.value if agent.constrained else agent.ceiling))


class ZeroIntelligenceSellerElement(Element):
    """Seller of one unit per trading period who asks a random price, above its cost under the budget constraint.

    In every round of a trading period, a seller that has not sold yet asks a price drawn uniformly from its cost, or
    from 0 without the constraint, to the price ceiling.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The cost of the unit to the seller.

        ask:
            The current ask.

        supply:
            The number of units still offered in the trading period.

        profit:
            The accumulated surplus of the seller's trades.

        ceiling:
            The highest price of the market.

        constrained:
            One when the seller never asks below its cost, zero otherwise.

    Examples:
        >>> from skcomplex.models.economics.markets import ZeroIntelligenceSellerElement
        >>> GroupElement('sellers', element=ZeroIntelligenceSellerElement, size=5, values={'value': 50.0}).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        ask: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 200.0,
        supply: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        profit: Annotated[float, StateSpec()] = 0.0,
        ceiling: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 200.0,
        constrained: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
        self.ask = ask
        self.supply = supply
        self.profit = profit
        self.ceiling = ceiling
        self.constrained = constrained

    @classmethod
    def _get_view(cls) -> type[ZeroIntelligenceSellerView]:
        """Return the view of a seller, with its cost, ask, supply, and profit."""
        return ZeroIntelligenceSellerView

    @staticmethod
    def get_ask(agent: ZeroIntelligenceSellerView, model: ModelView) -> float:
        """Return an ask drawn uniformly from the cost, or from 0 without the constraint, to the price ceiling.

        Args:
            agent:
                The seller.

            model:
                The model view of the quotes.

        Returns:
            ask:
                The ask of the round.
        """
        bottom = agent.value if agent.constrained else 0.0
        return float(model.rng.uniform(bottom, max(bottom, agent.ceiling)))


class AuctionLedgerInteraction(FeedbackInteraction[ModelView]):
    """Ledger that closes every trading period: it measures its surplus and gives every trader a new unit.

    The realised surplus is the value of the units bought minus the cost of the units sold, and the allocative
    efficiency is its ratio to the maximum surplus of the supply and demand schedules.

    Args:
        label:
            The label of the ledger.

        phase:
            The phase the ledger acts in.

    Examples:
        >>> from skcomplex.models.economics.markets import AuctionLedgerInteraction
        >>> AuctionLedgerInteraction('ledger', phase='close').reports
        ('surplus', 'efficiency')
    """

    reports: ClassVar[tuple[str, ...]] = ('surplus', 'efficiency')

    def act(self, model: ModelView) -> None:
        """Report the surplus and the efficiency of the trading period and open the next one.

        Args:
            model:
                The model view scoped to the ledger.
        """
        buyers = model.agents('buyers', ZeroIntelligenceBuyerView)
        sellers = model.agents('sellers', ZeroIntelligenceSellerView)
        values = np.array([buyer.value for buyer in buyers])
        costs = np.array([seller.value for seller in sellers])
        bought = np.array([buyer.demand <= 0 for buyer in buyers])
        sold = np.array([seller.supply <= 0 for seller in sellers])
        surplus = float(values[bought].sum() - costs[sold].sum())
        model.report('surplus', surplus)
        model.report('efficiency', surplus / maximum_surplus(values, costs))
        for buyer in buyers:
            buyer.demand = 1.0
        for seller in sellers:
            seller.supply = 1.0


@register_model(
    field='economics/markets',
    tags=('market', 'agents', 'double auction', 'zero intelligence'),
    references=('Gooding 2019 ch. 8-12', 'Gode and Sunder 1993'),
    defaults={'until': 20},
)
def double_auction(
    n_buyers: int = 50,
    n_sellers: int = 50,
    *,
    protocol: str = 'order_book',
    rounds: int = 20,
    constrained: bool = True,
    ceiling: float = 200.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Gode and Sunder's double auction of zero-intelligence traders, a step per trading period.

    Every buyer values one unit and every seller can produce one unit at a cost, both drawn uniformly below the price
    ceiling. During a trading period of several rounds, every trader that has not traded yet quotes a random price and
    the market matches the quotes, continuously in a limit order book or at one uniform price in a call auction. With
    the budget constraint (ZI-C) no trader accepts a loss, and the market alone reaches nearly the whole competitive
    surplus; without it (ZI-U), traders make losing trades and the efficiency drops.

    Args:
        n_buyers:
            The number of buyers.

        n_sellers:
            The number of sellers.

        protocol:
            The protocol of the market, `'order_book'` or `'call_auction'`.

        rounds:
            The number of quoting rounds of a trading period.

        constrained:
            Whether traders never quote at a loss.

        ceiling:
            The highest price of the market.

        random_state:
            The seed or generator of the values and the costs.

    Returns:
        system:
            The model, with the groups `buyers` and `sellers`, the market `auction`, and the realised `ledger.surplus`
            and `ledger.efficiency` of every trading period.

    Raises:
        ValueError:
            The protocol is unknown.

    Examples:
        >>> from skcomplex.models.economics.markets import double_auction
        >>> [group.size for group in double_auction(10, 8).elements]
        [10, 8]
    """
    if protocol not in PROTOCOLS:
        error_msg = f'Unknown protocol {protocol!r}. Use one of {PROTOCOLS}.'
        raise ValueError(error_msg)
    rng = np.random.default_rng(random_state)
    values = rng.uniform(0.25 * ceiling, ceiling, n_buyers)
    costs = rng.uniform(0.0, 0.75 * ceiling, n_sellers)
    limit = float(constrained)
    buyers = GroupElement(
        'buyers',
        element=ZeroIntelligenceBuyerElement,
        size=n_buyers,
        values={'value': values, 'ceiling': ceiling, 'constrained': limit},
    )
    sellers = GroupElement(
        'sellers',
        element=ZeroIntelligenceSellerElement,
        size=n_sellers,
        values={'value': costs, 'ceiling': ceiling, 'constrained': limit},
    )
    market = MatchingInteraction(
        'auction',
        requesters='buyers',
        providers='sellers',
        rule=ContinuousClearingSpec() if protocol == 'order_book' else BatchClearingSpec(),
        limit=('bid', 'ask'),
        quantity=('demand', 'supply'),
        phase='trade',
        on_match=_book_profit,
    )
    quotes = [
        MapInteraction(label, target=group, state=state, function=function, form='agent', phase='trade')
        for label, group, state, function in (
            ('bids', 'buyers', 'bid', ZeroIntelligenceBuyerElement.get_bid),
            ('asks', 'sellers', 'ask', ZeroIntelligenceSellerElement.get_ask),
        )
    ]
    return System(
        elements=[buyers, sellers],
        interactions=[*quotes, market, AuctionLedgerInteraction('ledger', phase='close')],
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('trade', repeat=rounds), PhaseSpec('close')])),
    )
