"""Markets models: shops on a grid that compete for nearby shoppers, and the digital market that removes distance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import MatchingInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import (
    GreedyMatchingSpec,
    MatchSpec,
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    StateSpec,
    VisualSpec,
)
from ..._catalog import register_model
from ._resident_view import ResidentView
from ._store_view import StoreView


def _largest_share(sales: object) -> float:
    """Return the share of the sales of the shop that sells the most."""
    values = np.asarray(sales, dtype=float)
    return float(values.max() / values.sum()) if values.size and values.sum() > 0 else 0.0


def _shops_observables(system: System) -> list[ObservableSpec]:
    """Return the number of shops, their mean price, and the share of the sales of the largest shop."""
    del system
    return [
        ObservableSpec('open_shops', group='shops', variable='price', reduce='count'),
        ObservableSpec('mean_price', group='shops', variable='price', reduce='mean'),
        ObservableSpec('concentration', group='shops', variable='sales', reduce=_largest_share),
    ]


class ResidentElement(Element):
    """Shopper who wants one unit a step and shops in person, paying for travel, or online, where distance is free.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.economics.markets import ResidentElement
        >>> GroupElement('shoppers', element=ResidentElement, size=3, values={'online': 1.0}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'travel', 'color_range': (0.0, 1.0)}),
        *,
        bid: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        money: Annotated[float, StateSpec()] = 1000.0,
        travel: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        distance_cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        online: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.bid = bid
        self.demand = demand
        self.money = money
        self.travel = travel
        self.distance_cost = distance_cost
        self.online = online

    @classmethod
    def _get_view(cls) -> type[ResidentView]:
        """Return the view of a shopper, with its states and shopping parameters."""
        return ResidentView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that decides how the shopper shops in the step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.plan, phase='market')]

    def plan(self, agent: ResidentView, model: ModelView) -> None:
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

    @staticmethod
    def get_preference(shopper: object, shop: object) -> float:
        """Return the preference of a shopper for a shop, the opposite of the delivered price.

        The delivered price is the shop's ask plus the shopper's travel cost per cell times the distance to the shop.

        Args:
            shopper:
                The shopper.

            shop:
                The shop.

        Returns:
            preference:
                The opposite of the delivered price, higher for a preferred shop.
        """
        if not isinstance(shopper, ResidentView) or not isinstance(shop, StoreView):
            return -math.inf
        return -(shop.ask + shopper.travel * shopper.distance_to(shop))


class StoreElement(Element):
    """Shop that offers its capacity at its price, searches for the price of the largest profit, and may close.

    The shop searches for its price by trial and error: it moves the price by `adjustment` in the same direction as
    long as its profit grows, and turns back when the profit falls, never below the unit cost. It pays the unit cost
    of every unit sold and the fixed cost every step, and closes when its money runs out.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.economics.markets import StoreElement
        >>> GroupElement('shops', element=StoreElement, size=4, values={'price': 10.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'price'}),
        *,
        price: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        ask: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        supply: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        sales: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        money: Annotated[float, StateSpec()] = 100.0,
        profit: Annotated[float, StateSpec()] = 0.0,
        direction: Annotated[float, StateSpec()] = 1.0,
        capacity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        unit_cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        fixed_cost: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        adjustment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.price = price
        self.ask = ask
        self.supply = supply
        self.sales = sales
        self.money = money
        self.profit = profit
        self.direction = direction
        self.capacity = capacity
        self.unit_cost = unit_cost
        self.fixed_cost = fixed_cost
        self.adjustment = adjustment

    @classmethod
    def _get_view(cls) -> type[StoreView]:
        """Return the view of a shop, with its states and cost parameters."""
        return StoreView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that opens the shop for the market and the rule that settles its accounts.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.open, phase='market'), RuleLaw(self.settle, phase='accounts')]

    def open(self, agent: StoreView, model: ModelView) -> None:
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

    def settle(self, agent: StoreView, model: ModelView) -> None:
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

    @staticmethod
    def record_sale(match: MatchSpec, model: ModelView) -> None:
        """Count the units of a purchase in the sales of the shop.

        Args:
            match:
                The purchase of a shopper, the requester, from a shop, the provider.

            model:
                The model view of the market.
        """
        shop = match.provider
        if isinstance(shop, StoreView):
            shop.sales += match.quantity


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

    Every step every shopper wants one unit and shops online with probability `online`, and in person otherwise, when it
    pays a travel cost for the distance to the shop. The shoppers choose in random order the shops they prefer by price
    plus travel cost, within the shops' capacities, a greedy matching that pays the shop's ask for a unit. Shops search
    for the price of the largest profit by trial and error, pay a fixed cost every step, and close when their money runs
    out. Distance gives every shop a local market, so many shops survive with high prices. In the digital market the
    cheapest shop takes every shopper, the others undercut it, prices fall toward the unit cost, and most shops close
    (Hamill and Gilbert 2016 sec. 6.3-6.4).

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
            The model, with the groups `shoppers` and `shops` and the market `shopping`, whose reporters are the
            `shopping.price`, `shopping.volume`, and `shopping.unmatched` of the purchases.

    Examples:
        >>> from skcomplex.models.economics.markets import shops
        >>> [group.size for group in shops(4, width=5, height=5).elements]
        [25, 4]
    """
    rng = np.random.default_rng(random_state)
    cells = [(x, y) for x in range(width) for y in range(height)]
    places = rng.choice(len(cells), size=min(n_shops, len(cells)), replace=False)
    shoppers = GroupElement(
        'shoppers',
        element=ResidentElement,
        size=len(cells),
        values={'bid': value, 'online': online, 'distance_cost': distance_cost},
        location=cells,
    )
    stores = GroupElement(
        'shops',
        element=StoreElement,
        size=len(places),
        values={
            'price': rng.uniform(0.25 * value, 0.75 * value, len(places)),
            'capacity': capacity,
            'fixed_cost': fixed_cost,
        },
        variable=True,
        location=[cells[index] for index in places],
    )
    shopping = MatchingInteraction[ModelView](
        'shopping',
        requesters='shoppers',
        providers='shops',
        rule=GreedyMatchingSpec(score=ResidentElement.get_preference),
        limit=('bid', 'ask'),
        quantity=('demand', 'supply'),
        transfers=(('money', -1),),
        phase='market',
        on_match=StoreElement.record_sale,
    )
    return System(
        elements=[shoppers, stores],
        interactions=[shopping],
        space=LatticeSpace((width, height), boundary='fixed', neighborhood='moore'),
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('market'), PhaseSpec('accounts')])),
    )
