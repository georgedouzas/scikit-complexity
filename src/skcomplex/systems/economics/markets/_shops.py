"""Markets systems: shops on a grid that compete for nearby shoppers.

It also holds the digital market that removes distance.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import (
    LifeCycleInteraction,
    MapInteraction,
    MatchingInteraction,
    SignAdjustmentInteraction,
)
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import (
    MatchSpec,
    ObservableSpec,
    ParameterSpec,
    PartnerSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    StateSpec,
    VisualSpec,
)


class ResidentView(AgentView):
    """Shopper as the rules see it.

    It holds its `bid`, `demand`, `money`, and `travel`, and its shopping parameters. The parameters are the
    `distance_cost` of shopping in person and the probability `online` of shopping online.
    """

    bid: float
    demand: float
    money: float
    travel: float
    distance_cost: float
    online: float


class StoreView(AgentView):
    """Shop as the rules see it.

    The states are the `price`, the `ask` of the step, the `supply` left, the `sales`, the `money`, the `profit`, and
    the `direction` of the last change of the price. The parameters are the `capacity`, the `unit_cost`, the
    `fixed_cost`, and the price `adjustment`.
    """

    price: float
    ask: float
    supply: float
    sales: float
    money: float
    profit: float
    direction: float
    capacity: float
    unit_cost: float
    fixed_cost: float
    adjustment: float


class ResidentElement(Element):
    """Shopper who wants one unit a step.

    It shops in person, paying for travel, or online, where distance is free.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import ResidentElement
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

    @staticmethod
    def get_travel(agent: ResidentView, system: SystemView) -> float:
        """Return the travel cost per cell of the step.

        It is zero when the shopper shops online with its probability.

        Args:
            agent:
                The shopper.

            system:
                The system view of the plan.

        Returns:
            travel:
                The travel cost per cell.
        """
        return 0.0 if system.rng.random() < agent.online else agent.distance_cost

    @staticmethod
    def get_preference(shopper: object, shop: object) -> float:
        """Return the preference of a shopper for a shop.

        It is the opposite of the delivered price. The delivered price is the shop's ask plus the shopper's travel cost
        per cell times the distance to the shop.

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
    """Shop that offers its capacity at its price.

    It searches for the price of the largest profit, and may close. The shop searches for its price by trial and error:
    it moves the price by `adjustment` in the same direction as long as its profit grows, and turns back when the profit
    falls, never below the unit cost. It pays the unit cost of every unit sold and the fixed cost every step, and closes
    when its money runs out.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import StoreElement
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

    @staticmethod
    def get_profit(agent: StoreView) -> float:
        """Return the profit of the step, the margin on the units sold less the fixed cost.

        Args:
            agent:
                The shop.

        Returns:
            profit:
                The profit of the step.
        """
        return agent.sales * (agent.ask - agent.unit_cost) - agent.fixed_cost

    @staticmethod
    def get_opening() -> tuple[tuple[str, Callable[[StoreView, SystemView], float]], ...]:
        """Return the states a shop sets when it opens for the market.

        It offers its capacity at its price.

        Returns:
            opening:
                The name of every state and the function of its value, of the shop and the system view.
        """
        return (
            ('supply', lambda agent, system: agent.capacity),
            ('ask', lambda agent, system: agent.price),
            ('sales', lambda agent, system: 0.0),
        )

    @staticmethod
    def get_settlement() -> tuple[tuple[str, Callable[[StoreView, SystemView], float]], ...]:
        """Return the states a shop sets when it settles its accounts, in their order.

        The shop turns its price back when its profit falls, records the profit, and pays its costs. The pricing of
        the system then steps the price by the sign of the direction.

        Returns:
            settlement:
                The name of every state and the function of its value, of the shop and the system view.
        """
        profit = StoreElement.get_profit
        return (
            ('direction', lambda agent, system: -agent.direction if profit(agent) < agent.profit else agent.direction),
            ('profit', lambda agent, system: profit(agent)),
            ('money', lambda agent, system: agent.money - agent.sales * agent.unit_cost - agent.fixed_cost),
        )

    @staticmethod
    def record_sale(match: MatchSpec, system: SystemView) -> None:
        """Count the units of a purchase in the sales of the shop.

        Args:
            match:
                The purchase of a shopper, the requester, from a shop, the provider.

            system:
                The system view of the market.
        """
        shop = match.provider
        if isinstance(shop, StoreView):
            shop.sales += match.quantity


def _largest_share(sales: object) -> float:
    """Return the share of the sales of the shop that sells the most."""
    values = np.asarray(sales, dtype=float)
    return float(values.max() / values.sum()) if values.size and values.sum() > 0 else 0.0


class ShopsSystem(BaseSystem):
    """Shops on a grid that compete for the shoppers on every cell.

    The shoppers buy in person or in a digital market. Every step every shopper wants one unit and shops online with
    probability `online`, and in person otherwise, when it pays a travel cost for the distance to the shop. The shoppers
    choose in random order the shops they prefer by price plus travel cost, within the shops' capacities, a greedy
    matching that pays the shop's ask for a unit. Shops search for the price of the largest profit by trial and error,
    pay a fixed cost every step, and close when their money runs out. Distance gives every shop a local market, so many
    shops survive with high prices. In the digital market the cheapest shop takes every shopper, the others undercut it,
    prices fall toward the unit cost, and most shops close (Hamill and Gilbert 2016 sec. 6.3-6.4).

    The system has the groups `shoppers` and `shops` and the market `shopping`, whose reporters are the
    `shopping.value`, `shopping.volume`, and `shopping.unmatched` of the purchases.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import ShopsSystem
        >>> [group.size for group in ShopsSystem(4, width=5, height=5).prepare().system_.elements]
        [25, 4]
    """

    field = 'economics/markets'
    topics = ('market', 'agents', 'retail', 'digital market', 'grid')
    references = ('Hamill and Gilbert 2016 sec. 6.3-6.4',)
    until = 100
    observables = (
        ObservableSpec('open_shops', group='shops', variable='price', reduce='count'),
        ObservableSpec('mean_price', group='shops', variable='price', reduce='mean'),
        ObservableSpec('concentration', group='shops', variable='sales', reduce=_largest_share),
    )

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_shops = n_shops
        self.width = width
        self.height = height
        self.online = online
        self.distance_cost = distance_cost
        self.capacity = capacity
        self.fixed_cost = fixed_cost
        self.value = value
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        places = rng.choice(len(cells), size=min(self.n_shops, len(cells)), replace=False)
        shoppers = GroupElement(
            'shoppers',
            element=ResidentElement,
            size=len(cells),
            values={'bid': self.value, 'online': self.online, 'distance_cost': self.distance_cost},
            location=cells,
        )
        stores = GroupElement(
            'shops',
            element=StoreElement,
            size=len(places),
            values={
                'price': rng.uniform(0.25 * self.value, 0.75 * self.value, len(places)),
                'capacity': self.capacity,
                'fixed_cost': self.fixed_cost,
            },
            variable=True,
            location=[cells[index] for index in places],
        )
        plan = [
            MapInteraction(
                'demand',
                target='shoppers',
                state='demand',
                function=lambda agent, system: 1.0,
                form='agent',
                phase='market',
            ),
            MapInteraction(
                'travel',
                target='shoppers',
                state='travel',
                function=ResidentElement.get_travel,
                form='agent',
                phase='market',
            ),
            *(
                MapInteraction(
                    f'opening_{name}',
                    target='shops',
                    state=name,
                    function=function,
                    form='agent',
                    phase='market',
                )
                for name, function in StoreElement.get_opening()
            ),
        ]
        settlement = [
            *(
                MapInteraction(
                    f'settlement_{name}',
                    target='shops',
                    state=name,
                    function=function,
                    form='agent',
                    phase='accounts',
                )
                for name, function in StoreElement.get_settlement()
            ),
            SignAdjustmentInteraction(
                'settlement_price',
                target='shops',
                state='price',
                signal=lambda agent, system: agent.direction,
                step=lambda agent, system: agent.adjustment,
                lower=lambda agent, system: agent.unit_cost,
                form='agent',
                phase='accounts',
            ),
            LifeCycleInteraction(
                'closure',
                target='shops',
                death=lambda agent, system: agent.money < 0,
                phase='accounts',
            ),
        ]
        shopping = MatchingInteraction[SystemView](
            'shopping',
            requesters='shoppers',
            providers='shops',
            rule=PartnerSpec(choice='score', score=ResidentElement.get_preference, clearing='provider'),
            limit=('bid', 'ask'),
            quantity=('demand', 'supply'),
            transfers=(('money', -1),),
            phase='market',
            on_match=StoreElement.record_sale,
        )
        return System(
            self.label,
            elements=[shoppers, stores],
            interactions=[*plan, shopping, *settlement],
            space=LatticeSpace((self.width, self.height), boundary='fixed', metric='chebyshev'),
            time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('market'), PhaseSpec('accounts')])),
        )
