"""Markets models: shops on a grid that compete for nearby shoppers, and the digital market that removes distance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction, Match, matches
from ....core.laws import Law, Rule, Symbols, transfer
from ....core.runtime import register_model
from ....core.space import DiscreteTime, LatticeSpace, Phase, Phases
from ....core.system import Observable, System
from ..realizations import Trade


def _largest_share(sales: Any) -> float:  # noqa: ANN401
    """Return the share of the sales of the shop that sells the most."""
    values = np.asarray(sales, dtype=float)
    return float(values.max() / values.sum()) if values.size and values.sum() > 0 else 0.0


def _shops_observables(system: System) -> list[Observable]:
    """Return the number of shops, their mean price, and the share of the sales of the largest shop."""
    del system
    return [
        Observable('open_shops', group='shops', variable='price', reduce='count'),
        Observable('mean_price', group='shops', variable='price', reduce='mean'),
        Observable('concentration', group='shops', variable='sales', reduce=_largest_share),
    ]


@dataclass(frozen=True)
class DeliveredPriceChoice:
    """Pairing of shoppers, in random order, with the shop of the lowest delivered price that has units left.

    The delivered price is the shop's ask plus the shopper's travel cost per cell times the distance to the shop. A
    shopper buys one unit when the ask is at most its bid, so every pair is a trade of one unit at the ask. The
    shoppers choose in turn, so the matching is stable when shops do not choose among shoppers.

    Examples:
        >>> from skcomplex.models.economics.markets import DeliveredPriceChoice
        >>> DeliveredPriceChoice()
        DeliveredPriceChoice()
    """

    def __call__(
        self,
        left: Sequence[Any],
        right: Sequence[Any],
        rng: np.random.Generator,
        capacity: int,
    ) -> list[Any]:
        """Return the purchases of the shoppers, as a pairing of the matching law.

        Args:
            left:
                The shoppers.

            right:
                The shops.

            rng:
                The random generator.

            capacity:
                The largest number of purchases of every shopper, which the demand limits instead.

        Returns:
            trades:
                The purchases.
        """
        shoppers = [agent for agent in left if agent.demand > 0]
        stores = [agent for agent in right if agent.supply > 0]
        if not shoppers or not stores:
            return []
        here = np.array([agent.location for agent in shoppers], dtype=float)
        there = np.array([agent.location for agent in stores], dtype=float)
        distances = np.linalg.norm(here[:, None, :] - there[None, :, :], axis=-1)
        asks = np.array([agent.ask for agent in stores], dtype=float)
        travel = np.array([agent.travel for agent in shoppers], dtype=float)
        delivered = asks[None, :] + travel[:, None] * distances
        supply = np.array([agent.supply for agent in stores], dtype=float)
        trades = []
        for index in rng.permutation(len(shoppers)):
            shopper = shoppers[int(index)]
            costs = np.where((supply >= 1) & (asks <= shopper.bid), delivered[index], np.inf)
            best = int(np.argmin(costs))
            if not np.isfinite(costs[best]):
                continue
            supply[best] -= 1
            trades.append(Trade(shopper, stores[best], 1.0, float(asks[best])))
        return trades


class Resident(Element):
    """Shopper who wants one unit a step and shops in person, paying for travel, or online, where distance is free.

    Args:
        label:
            The label of the agent or group.

        bid:
            The highest price the shopper pays.

        demand:
            The units still wanted in the step.

        money:
            The money of the shopper.

        travel:
            The travel cost per cell of the current step, zero online.

        distance_cost:
            The travel cost per cell of shopping in person.

        online:
            The probability of shopping online in a step.

    Examples:
        >>> from skcomplex.models.economics.markets import Resident
        >>> Resident.group('shoppers', 3, online=1.0).size
        3
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'travel', 'color_range': (0.0, 1.0)})
    bid: float = state(default=20.0, domain=Domain.NONNEGATIVE)
    demand: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    money: float = state(default=1000.0, domain=Domain.REAL)
    travel: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    distance_cost: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    online: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule that decides how the shopper shops in the step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.plan, phase='market')]

    def plan(self, agent: AgentView, model: ModelView) -> None:
        """Want one unit, and shop online with the probability of shopping online.

        Args:
            agent:
                The shopper.

            model:
                The model.
        """
        agent.demand = 1.0
        online = model.rng.random() < model.parameter('online')
        agent.travel = 0.0 if online else model.parameter('distance_cost')


class Store(Element):
    """Shop that offers its capacity at its price, searches for the price of the largest profit, and may close.

    The shop searches for its price by trial and error: it moves the price by `adjustment` in the same direction as
    long as its profit grows, and turns back when the profit falls, never below the unit cost. It pays the unit cost
    of every unit sold and the fixed cost every step, and closes when its money runs out.

    Args:
        label:
            The label of the agent or group.

        price:
            The price of a unit.

        ask:
            The price offered in the current step.

        supply:
            The units still offered in the step.

        sales:
            The units sold in the last step.

        money:
            The money of the shop.

        profit:
            The profit of the last step.

        direction:
            The direction of the last change of the price, 1 up or -1 down.

        capacity:
            The units the shop can sell in a step.

        unit_cost:
            The cost of a unit sold.

        fixed_cost:
            The cost of running the shop for a step.

        adjustment:
            The relative change of the price in a step.

    Examples:
        >>> from skcomplex.models.economics.markets import Store
        >>> Store.group('shops', 4, price=10.0).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'price'})
    price: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    ask: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    supply: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    sales: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    money: float = state(default=100.0, domain=Domain.REAL)
    profit: float = state(default=0.0, domain=Domain.REAL)
    direction: float = state(default=1.0, domain=Domain.REAL)
    capacity: float = parameter(default=1000.0, domain=Domain.POSITIVE)
    unit_cost: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    fixed_cost: float = parameter(default=20.0, domain=Domain.NONNEGATIVE)
    adjustment: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule that opens the shop for the market and the rule that settles its accounts.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.open, phase='market'), Rule(self.settle, phase='accounts')]

    def open(self, agent: AgentView, model: ModelView) -> None:
        """Offer the capacity at the price.

        Args:
            agent:
                The shop.

            model:
                The model.
        """
        agent.supply = model.parameter('capacity')
        agent.ask = agent.price
        agent.sales = 0.0

    def settle(self, agent: AgentView, model: ModelView) -> None:
        """Pay the costs, close when the money runs out, and move the price toward a larger profit.

        Args:
            agent:
                The shop.

            model:
                The model.
        """
        unit_cost = model.parameter('unit_cost')
        profit = agent.sales * (agent.ask - unit_cost) - model.parameter('fixed_cost')
        agent.money -= agent.sales * unit_cost + model.parameter('fixed_cost')
        if agent.money < 0:
            agent.remove()
            return
        if profit < agent.profit:
            agent.direction = -agent.direction
        agent.profit = profit
        agent.price = max(unit_cost, agent.price * (1 + agent.direction * model.parameter('adjustment')))


class Shopping(Interaction):
    """Market where shoppers buy from the shops they prefer, by price plus travel cost.

    The matching law pairs every shopper with the shop of the lowest price plus travel cost that has units left, and
    every pair trades one unit at the shop's price, paid as a transfer from shopper to shop.

    Args:
        label:
            The label of the market.

        shoppers:
            The path of the group of shoppers.

        shops:
            The path of the group of shops.

    Examples:
        >>> from skcomplex.models.economics.markets import Shopping
        >>> Shopping('shopping', shoppers='shoppers', shops='shops').reports
        ('volume',)
    """

    reports: ClassVar[tuple[str, ...]] = ('volume',)
    shoppers: str = role()
    shops: str = role()

    def laws(self, s: Symbols) -> list[Any]:
        """Return the matching of shoppers with shops and the rule that settles the trades.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Match('shoppers', 'shops', pairing=DeliveredPriceChoice(), label='purchases', phase='market'),
            Rule(self.buy, phase='market'),
        ]

    def buy(self, model: Any) -> None:  # noqa: ANN401
        """Pay for every purchase and count the units sold.

        Args:
            model:
                The model view of the market.
        """
        volume = 0.0
        for trade in cast('list[Trade]', matches(model, 'purchases')):
            transfer(trade.buyer, trade.seller, trade.quantity * trade.price, 'money')
            trade.buyer.demand -= trade.quantity
            trade.seller.supply -= trade.quantity
            trade.seller.sales += trade.quantity
            volume += trade.quantity
        model.report('volume', volume)


@register_model(
    field='economics/markets',
    tags=('market', 'agents', 'retail', 'digital market', 'grid'),
    references=('Hamill and Gilbert 2016 sec. 6.3-6.4',),
    defaults={'until': 100},
    observables=_shops_observables,
)
def shops(
    n_shops: int = 16,
    *,
    width: int = 20,
    height: int = 20,
    online: float = 0.0,
    distance_cost: float = 1.0,
    capacity: float = 1000.0,
    fixed_cost: float = 20.0,
    value: float = 20.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return shops on a grid that compete for the shoppers on every cell, in person or in a digital market.

    Every step every shopper wants one unit and shops online with probability `online`, and in person otherwise, when
    it pays a travel cost for the distance to the shop. The shoppers choose in random order the shops they prefer
    by price plus travel cost, within the shops' capacities. Shops search for the price of the largest profit by trial
    and error, pay a fixed cost every step, and close when their money runs out. Distance gives every shop a local
    market, so many shops survive with high prices. In the digital market the cheapest shop takes every shopper, the
    others undercut it, prices fall toward the unit cost, and most shops close (Hamill and Gilbert 2016 sec.
    6.3-6.4).

    Args:
        n_shops:
            The initial number of shops.

        width:
            The number of columns of the grid, with one shopper per cell.

        height:
            The number of rows of the grid.

        online:
            The probability that a shopper shops online in a step.

        distance_cost:
            The travel cost per cell of shopping in person.

        capacity:
            The units every shop can sell in a step.

        fixed_cost:
            The cost of running a shop for a step.

        value:
            The highest price a shopper pays.

        random_state:
            The seed or generator of the cells and initial prices of the shops.

    Returns:
        system:
            The model, with the groups `shoppers` and `shops` and the market `shopping`.

    Examples:
        >>> from skcomplex.models.economics.markets import shops
        >>> [group.size for group in shops(4, width=5, height=5).elements]
        [25, 4]
    """
    rng = np.random.default_rng(random_state)
    cells = [(x, y) for x in range(width) for y in range(height)]
    places = rng.choice(len(cells), size=min(n_shops, len(cells)), replace=False)
    shoppers = Resident.group(
        'shoppers',
        len(cells),
        location=cells,
        bid=value,
        online=online,
        distance_cost=distance_cost,
    )
    stores = Store.group(
        'shops',
        len(places),
        variable=True,
        location=[cells[index] for index in places],
        price=rng.uniform(0.25 * value, 0.75 * value, len(places)),
        capacity=capacity,
        fixed_cost=fixed_cost,
    )
    return System(
        components=[shoppers, stores],
        interactions=[Shopping('shopping', shoppers='shoppers', shops='shops')],
        space=LatticeSpace((width, height), boundary='fixed', neighborhood='moore'),
        time=DiscreteTime(schedule=Phases([Phase('market'), Phase('accounts')])),
    )
