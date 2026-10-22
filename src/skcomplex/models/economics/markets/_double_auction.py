"""Markets models: double auctions of zero-intelligence traders with and without budget constraints."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.base import Domain, parameter, state
from ....core.model import (
    AgentType,
    CallAuction,
    DiscreteTime,
    Institution,
    Law,
    Market,
    OrderBook,
    Phase,
    Phases,
    Rule,
    Symbols,
    System,
    Trade,
)
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

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


def _book_profit(trade: Trade, model: ModelView) -> None:
    """Add the surplus of a trade to the profits of the buyer and the seller."""
    del model
    trade.buyer.profit += trade.quantity * (trade.buyer.value - trade.price)
    trade.seller.profit += trade.quantity * (trade.price - trade.seller.value)


class ZeroIntelligenceBuyer(AgentType):
    """Buyer of one unit per trading period who bids a random price, below its value under the budget constraint.

    In every round of a trading period, a buyer that has not bought yet bids a price drawn uniformly from 0 to its
    value, or to the price ceiling without the constraint, the zero-intelligence traders of Gode and Sunder (1993).

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

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
        >>> from skcomplex.models.economics.markets import ZeroIntelligenceBuyer
        >>> ZeroIntelligenceBuyer.group('buyers', 5, value=100.0).size
        5
    """

    value: float = state(default=100.0, domain=Domain.NONNEGATIVE)
    bid: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    demand: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    profit: float = state(default=0.0, domain=Domain.REAL)
    ceiling: float = parameter(default=200.0, domain=Domain.POSITIVE)
    constrained: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the bidding rule of the trading rounds.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.quote, phase='trade')]

    def quote(self, agent: AgentView, model: ModelView) -> None:
        """Bid a random price.

        Args:
            agent:
                The buyer.

            model:
                The model.
        """
        top = agent.value if model.parameter('constrained') else model.parameter('ceiling')
        agent.bid = float(model.rng.uniform(0.0, top))


class ZeroIntelligenceSeller(AgentType):
    """Seller of one unit per trading period who asks a random price, above its cost under the budget constraint.

    In every round of a trading period, a seller that has not sold yet asks a price drawn uniformly from its cost, or
    from 0 without the constraint, to the price ceiling.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

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
        >>> from skcomplex.models.economics.markets import ZeroIntelligenceSeller
        >>> ZeroIntelligenceSeller.group('sellers', 5, value=50.0).size
        5
    """

    value: float = state(default=100.0, domain=Domain.NONNEGATIVE)
    ask: float = state(default=200.0, domain=Domain.NONNEGATIVE)
    supply: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    profit: float = state(default=0.0, domain=Domain.REAL)
    ceiling: float = parameter(default=200.0, domain=Domain.POSITIVE)
    constrained: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the asking rule of the trading rounds.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.quote, phase='trade')]

    def quote(self, agent: AgentView, model: ModelView) -> None:
        """Ask a random price.

        Args:
            agent:
                The seller.

            model:
                The model.
        """
        bottom = agent.value if model.parameter('constrained') else 0.0
        agent.ask = float(model.rng.uniform(bottom, max(bottom, model.parameter('ceiling'))))


class AuctionLedger(Institution):
    """Ledger that closes every trading period: it measures its surplus and gives every trader a new unit.

    The realised surplus is the value of the units bought minus the cost of the units sold, and the allocative
    efficiency is its ratio to the maximum surplus of the supply and demand schedules.

    Args:
        label:
            The label of the ledger.

        phase:
            The phase the ledger acts in.

    Examples:
        >>> from skcomplex.models.economics.markets import AuctionLedger
        >>> AuctionLedger('ledger', phase='close').reports
        ('surplus', 'efficiency')
    """

    reports: ClassVar[tuple[str, ...]] = ('surplus', 'efficiency')

    def act(self, model: ModelView) -> None:
        """Report the surplus and the efficiency of the trading period and open the next one.

        Args:
            model:
                The model view scoped to the ledger.
        """
        buyers, sellers = model.agents('buyers'), model.agents('sellers')
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
    buyers = ZeroIntelligenceBuyer.group('buyers', n_buyers, value=values, ceiling=ceiling, constrained=limit)
    sellers = ZeroIntelligenceSeller.group('sellers', n_sellers, value=costs, ceiling=ceiling, constrained=limit)
    market = Market(
        'auction',
        buyers='buyers',
        sellers='sellers',
        protocol=OrderBook() if protocol == 'order_book' else CallAuction(),
        phase='trade',
        money=None,
        goods=None,
        on_trade=_book_profit,
    )
    return System(
        components=[buyers, sellers],
        interactions=[market, AuctionLedger('ledger', phase='close')],
        time=DiscreteTime(schedule=Phases([Phase('trade', repeat=rounds), Phase('close')])),
    )
