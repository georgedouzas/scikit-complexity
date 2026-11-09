"""Markets systems: double auctions of zero-intelligence traders.

The traders trade with and without budget constraints.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import FeedbackInteraction, MapInteraction, MatchingInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import (
    BatchMatchingSpec,
    ContinuousMatchingSpec,
    MatchSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    StateSpec,
    VisualSpec,
)


class ZeroIntelligenceBuyerView(AgentView):
    """Zero-intelligence buyer as the rules see it.

    It reads its `value`, `bid`, `demand`, and `profit`, the price `ceiling`, and whether it is `constrained`.
    """

    value: float
    bid: float
    demand: float
    profit: float
    ceiling: float
    constrained: float


class ZeroIntelligenceBuyerElement(Element):
    """Buyer of one unit per trading period who bids a random price.

    Under the budget constraint, the bid is below its value. In every round of a trading period, a buyer that has not
    bought yet bids a price drawn uniformly from 0 to its value, or to the price ceiling without the constraint, the
    zero-intelligence traders of Gode and Sunder (1993).

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import ZeroIntelligenceBuyerElement
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
    def get_bid(agent: ZeroIntelligenceBuyerView, system: SystemView) -> float:
        """Return a bid drawn uniformly from 0 to the value.

        Without the constraint, the bid goes up to the price ceiling.

        Args:
            agent:
                The buyer.

            system:
                The system view of the quotes.

        Returns:
            bid:
                The bid of the round.
        """
        return float(system.rng.uniform(0.0, agent.value if agent.constrained else agent.ceiling))


class ZeroIntelligenceSellerView(AgentView):
    """Zero-intelligence seller as the rules see it.

    It reads its cost `value`, `ask`, `supply`, and `profit`, the price `ceiling`, and whether it is `constrained`.
    """

    value: float
    ask: float
    supply: float
    profit: float
    ceiling: float
    constrained: float


class ZeroIntelligenceSellerElement(Element):
    """Seller of one unit per trading period who asks a random price.

    Under the budget constraint, the ask is above its cost. In every round of a trading period, a seller that has not
    sold yet asks a price drawn uniformly from its cost, or from 0 without the constraint, to the price ceiling.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import ZeroIntelligenceSellerElement
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
    def get_ask(agent: ZeroIntelligenceSellerView, system: SystemView) -> float:
        """Return an ask drawn uniformly from the cost to the price ceiling.

        Without the constraint, the ask starts from 0.

        Args:
            agent:
                The seller.

            system:
                The system view of the quotes.

        Returns:
            ask:
                The ask of the round.
        """
        bottom = agent.value if agent.constrained else 0.0
        return float(system.rng.uniform(bottom, max(bottom, agent.ceiling)))


def maximum_surplus(values: np.ndarray, costs: np.ndarray) -> float:
    """Return the largest total surplus of single units.

    It is the competitive surplus of the supply and demand schedules.

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
        >>> from skcomplex.systems.economics.markets import maximum_surplus
        >>> maximum_surplus(np.array([10.0, 6.0, 2.0]), np.array([1.0, 4.0, 8.0]))
        11.0
    """
    demand, supply = np.sort(values)[::-1], np.sort(costs)
    size = min(len(demand), len(supply))
    return float(np.maximum(demand[:size] - supply[:size], 0).sum())


class AuctionLedgerInteraction(FeedbackInteraction[SystemView]):
    """Ledger that closes every trading period.

    It measures its surplus and gives every trader a new unit. The realised surplus is the value of the units bought
    minus the cost of the units sold, and the allocative efficiency is its ratio to the maximum surplus of the supply
    and demand schedules.

    Args:
        label:
            The label of the ledger.

        phase:
            The phase the ledger acts in.

    Examples:
        >>> from skcomplex.systems.economics.markets import AuctionLedgerInteraction
        >>> AuctionLedgerInteraction('ledger', phase='close').reports
        ('surplus', 'efficiency')
    """

    reports: ClassVar[tuple[str, ...]] = ('surplus', 'efficiency')

    def act(self, system: SystemView) -> None:
        """Report the surplus and the efficiency of the trading period.

        It then opens the next one.

        Args:
            system:
                The system view scoped to the ledger.
        """
        buyers = system.agents('buyers', ZeroIntelligenceBuyerView)
        sellers = system.agents('sellers', ZeroIntelligenceSellerView)
        values = np.array([buyer.value for buyer in buyers])
        costs = np.array([seller.value for seller in sellers])
        bought = np.array([buyer.demand <= 0 for buyer in buyers])
        sold = np.array([seller.supply <= 0 for seller in sellers])
        surplus = float(values[bought].sum() - costs[sold].sum())
        system.report('surplus', surplus)
        system.report('efficiency', surplus / maximum_surplus(values, costs))
        for buyer in buyers:
            buyer.demand = 1.0
        for seller in sellers:
            seller.supply = 1.0


PROTOCOLS = ('order_book', 'call_auction')


class DoubleAuctionSystem(BaseSystem):
    """Gode and Sunder's double auction of zero-intelligence traders.

    Each step is a trading period. Every buyer values one unit and every seller can produce one unit at a cost, both
    drawn uniformly below the price ceiling. During a trading period of several rounds, every trader that has not traded
    yet quotes a random price and the market matches the quotes, continuously in a limit order book or at one uniform
    price in a call auction. With the budget constraint (ZI-C) no trader accepts a loss, and the market alone reaches
    nearly the whole competitive surplus. Without it (ZI-U), traders make losing trades and the efficiency drops.

    The traders are the zero-intelligence traders of Gode and Sunder (1993). The double auction of Gooding (2019), whose
    traders adapt their bids and asks, is a different model. The system has the groups `buyers` and `sellers`, the
    market `auction`, and the realised `ledger.surplus` and `ledger.efficiency` of every trading period.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValueError:
            The protocol is unknown.

    Examples:
        >>> from skcomplex.systems.economics.markets import DoubleAuctionSystem
        >>> [group.size for group in DoubleAuctionSystem(10, 8).prepare().system_.elements]
        [10, 8]
    """

    field = 'economics/markets'
    topics = ('market', 'agents', 'double auction', 'zero intelligence')
    references = ('Gode and Sunder 1993',)
    until = 20

    def __init__(
        self,
        n_buyers: int = 50,
        n_sellers: int = 50,
        *,
        protocol: str = 'order_book',
        rounds: int = 20,
        constrained: bool = True,
        ceiling: float = 200.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_buyers = n_buyers
        self.n_sellers = n_sellers
        self.protocol = protocol
        self.rounds = rounds
        self.constrained = constrained
        self.ceiling = ceiling
        self.random_state = random_state

    @staticmethod
    def _book_profit(trade: MatchSpec, system: SystemView) -> None:
        """Add the surplus of a trade to the profits of the buyer and the seller."""
        del system
        buyer, seller = trade.requester, trade.provider
        if not isinstance(buyer, ZeroIntelligenceBuyerView) or not isinstance(seller, ZeroIntelligenceSellerView):
            return
        buyer.profit += trade.quantity * (buyer.value - trade.value)
        seller.profit += trade.quantity * (trade.value - seller.value)

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        if self.protocol not in PROTOCOLS:
            error_msg = f'Unknown protocol {self.protocol!r}. Use one of {PROTOCOLS}.'
            raise ValueError(error_msg)
        rng = np.random.default_rng(self.random_state)
        values = rng.uniform(0.25 * self.ceiling, self.ceiling, self.n_buyers)
        costs = rng.uniform(0.0, 0.75 * self.ceiling, self.n_sellers)
        limit = float(self.constrained)
        buyers = GroupElement(
            'buyers',
            element=ZeroIntelligenceBuyerElement,
            size=self.n_buyers,
            values={'value': values, 'ceiling': self.ceiling, 'constrained': limit},
        )
        sellers = GroupElement(
            'sellers',
            element=ZeroIntelligenceSellerElement,
            size=self.n_sellers,
            values={'value': costs, 'ceiling': self.ceiling, 'constrained': limit},
        )
        market = MatchingInteraction(
            'auction',
            requesters='buyers',
            providers='sellers',
            rule=ContinuousMatchingSpec() if self.protocol == 'order_book' else BatchMatchingSpec(),
            reserve=('bid', 'ask'),
            quantity=('demand', 'supply'),
            phase='trade',
            on_match=self._book_profit,
        )
        quotes = [
            MapInteraction(label, target=group, state=state, function=function, form='agent', phase='trade')
            for label, group, state, function in (
                ('bids', 'buyers', 'bid', ZeroIntelligenceBuyerElement.get_bid),
                ('asks', 'sellers', 'ask', ZeroIntelligenceSellerElement.get_ask),
            )
        ]
        return System(
            self.label,
            elements=[buyers, sellers],
            interactions=[*quotes, market, AuctionLedgerInteraction('ledger', phase='close')],
            time=DiscreteTime(
                schedule=PhasesScheduleSpec([PhaseSpec('trade', repeat=self.rounds), PhaseSpec('close')]),
            ),
        )
