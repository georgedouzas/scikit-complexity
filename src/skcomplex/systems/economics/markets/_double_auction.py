"""Markets systems: double auctions of zero-intelligence traders.

The traders trade with and without budget constraints.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.matching import BatchMatchingSpec, ContinuousMatchingSpec, MatchingInteraction, MatchSpec
from ..common.policy import PolicyRuleInteraction

PROTOCOLS = ('order_book', 'call_auction')


@runtime_checkable
class ZeroIntelligenceBuyerView(AgentViewProtocol, Protocol):
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

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import ZeroIntelligenceBuyerElement
        >>> ZeroIntelligenceBuyerElement('buyers', value=100.0, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        *,
        value: Annotated[object, RealSet.closed_open(0, oo)] = 100.0,
        bid: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        demand: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        profit: object = 0.0,
        ceiling: Annotated[object, RealSet.open(0, oo)] = 200.0,
        constrained: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.value = value
        self.bid = bid
        self.demand = demand
        self.profit = profit
        self.ceiling = ceiling
        self.constrained = constrained


@runtime_checkable
class ZeroIntelligenceSellerView(AgentViewProtocol, Protocol):
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

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import ZeroIntelligenceSellerElement
        >>> ZeroIntelligenceSellerElement('sellers', value=50.0, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        *,
        value: Annotated[object, RealSet.closed_open(0, oo)] = 100.0,
        ask: Annotated[object, RealSet.closed_open(0, oo)] = 200.0,
        supply: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        profit: object = 0.0,
        ceiling: Annotated[object, RealSet.open(0, oo)] = 200.0,
        constrained: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.value = value
        self.ask = ask
        self.supply = supply
        self.profit = profit
        self.ceiling = ceiling
        self.constrained = constrained


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


class QuotingInteraction(Interaction):
    """Quotes of the zero-intelligence traders in every round.

    Every buyer bids a price drawn uniformly from 0 to its value and every seller asks a price drawn uniformly from its
    cost to the price ceiling. Without the budget constraint, the bid goes up to the ceiling and the ask starts from 0.

    Args:
        label:
            The label of the quotes.

        buyers:
            The path of the group of buyers.

        sellers:
            The path of the group of sellers.

        phase:
            The phase the traders quote in.

    Examples:
        >>> from skcomplex.systems.economics.markets import QuotingInteraction
        >>> QuotingInteraction('quotes', buyers='buyers', sellers='sellers').phase
        'trade'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, buyers: Role, sellers: Role, phase: str | None = 'trade') -> None:
        super().__init__(label)
        self.buyers = buyers
        self.sellers = sellers
        self.phase = phase

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the quotes of every round as the rule of the system."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context, self.quote, phase=self.phase)]

    def quote(self, system: SystemView) -> None:
        """Draw the bid of every buyer and then the ask of every seller.

        Args:
            system:
                The system view of the quotes.
        """
        for buyer in cast(list[ZeroIntelligenceBuyerView], system.get_agents('buyers')):
            buyer.bid = float(system.rng.uniform(0.0, buyer.value if buyer.constrained else buyer.ceiling))
        for seller in cast(list[ZeroIntelligenceSellerView], system.get_agents('sellers')):
            bottom = seller.value if seller.constrained else 0.0
            seller.ask = float(system.rng.uniform(bottom, max(bottom, seller.ceiling)))


class AuctionLedgerInteraction(PolicyRuleInteraction[SystemView]):
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
        buyers = cast(list[ZeroIntelligenceBuyerView], system.get_agents('buyers'))
        sellers = cast(list[ZeroIntelligenceSellerView], system.get_agents('sellers'))
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_buyers = n_buyers
        self.n_sellers = n_sellers
        self.protocol = protocol
        self.rounds = rounds
        self.constrained = constrained
        self.ceiling = ceiling
        self.random_state = random_state

    @staticmethod
    def _record_profit(trade: MatchSpec, system: SystemView) -> None:
        """Add the surplus of a trade to the profits of the buyer and the seller."""
        del system
        buyer, seller = trade.requester, trade.provider
        if not isinstance(buyer, ZeroIntelligenceBuyerView) or not isinstance(seller, ZeroIntelligenceSellerView):
            return
        buyer.profit += trade.quantity * (buyer.value - trade.value)
        seller.profit += trade.quantity * (trade.value - seller.value)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        if self.protocol not in PROTOCOLS:
            error_msg = f'Unknown protocol {self.protocol!r}. Use one of {PROTOCOLS}.'
            raise ValueError(error_msg)
        rng = np.random.default_rng(self.random_state)
        values = rng.uniform(0.25 * self.ceiling, self.ceiling, self.n_buyers)
        costs = rng.uniform(0.0, 0.75 * self.ceiling, self.n_sellers)
        limit = float(self.constrained)
        buyers = ZeroIntelligenceBuyerElement(
            'buyers',
            value=values,
            ceiling=self.ceiling,
            constrained=limit,
            size=self.n_buyers,
        )
        sellers = ZeroIntelligenceSellerElement(
            'sellers',
            value=costs,
            ceiling=self.ceiling,
            constrained=limit,
            size=self.n_sellers,
        )
        market = MatchingInteraction(
            'auction',
            requesters='buyers',
            providers='sellers',
            rule=ContinuousMatchingSpec() if self.protocol == 'order_book' else BatchMatchingSpec(),
            reserve=('bid', 'ask'),
            quantity=('demand', 'supply'),
            phase='trade',
            on_match=self._record_profit,
        )
        self.system_ = System(
            self.label,
            elements=[buyers, sellers],
            interactions=[
                QuotingInteraction('quotes', buyers='buyers', sellers='sellers'),
                market,
                AuctionLedgerInteraction('ledger', phase='close'),
            ],
            solver=MesaSolver(phases=[PhaseSpec('trade', repeat=self.rounds), PhaseSpec('close')]),
        )
