"""Markets systems: shops on a grid that compete for nearby shoppers.

It also holds the digital market that removes distance.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, SupportsFloat, cast

import numpy as np
from sage.all__sagemath_symbolics import SR, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import clip, sign
from ....core.components.spaces import BaseSpace, DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ValidationError
from ....core.specs import (
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    StateSpec,
    VisualSpec,
)
from ...general.agents import BirthDeathInteraction
from ...general.matching import MatchingInteraction, MatchSpec, PartnerMatchingSpec

STEPPINGS = ('multiplicative', 'additive')


class PriceAdjustmentInteraction(MapInteraction):
    """Price adjustment of a state of discrete time by the sign of a signal, kept within bounds.

    Economics names it a price adjustment by trial and error, as a shop that raises its price while demand exceeds its
    supply and lowers it otherwise. With the stepping `'multiplicative'` the next value of the state `x` is
    `x (1 + step sign(signal))`, and with the stepping `'additive'` it is `x + step sign(signal)`, so the state rises
    while the signal is positive, falls while it is negative, and stays while it is zero, by trial and error rather
    than by the size of the signal. The next value is then limited to `[lower, upper]`.

    In the `'symbolic'` form, the signal, the step, and the bounds are numbers or functions of the namespace, where
    `namespace.target` holds the states and parameters of the target. In the `'agent'` form, the target is an agent
    group and they are numbers or functions `f(agent, system)` that return numbers. The agents step under the schedules
    of `MapInteraction`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the element or group whose state steps.

        state:
            The name of the state of the target.

        signal:
            The function of the signal whose sign sets the direction of the step.

        step:
            The size of the step, relative with the stepping `'multiplicative'`, a number or a function.

        stepping:
            `'multiplicative'` or `'additive'`.

        lower:
            The lowest next value, a number or a function, or `None` for none.

        upper:
            The highest next value, a number or a function, or `None` for none.

        form:
            `'symbolic'` for an update law of the state of the target, or `'agent'` for a rule over an agent group.

        phase:
            The phase of the step the agent form runs in, or `None` for the default phase.

        schedule:
            The order of the agents in the agent form, `'synchronous'`, `'sequential'`, `'random-sequential'`, or
            `'member'`.

    Raises:
        ValidationError:
            The stepping, the form, or the schedule is unknown, the symbolic form is not synchronous, or the signal is
            not callable, raised by `prepare`.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.spaces import DiscreteTime
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.economics.markets import PriceAdjustmentInteraction
        >>> class QuoteElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        >>> search = PriceAdjustmentInteraction(
        ...     'search', target='quote', state='x', signal=lambda s: 2 - s.target.x, step=0.5, upper=2.0
        ... )
        >>> system = CompositeSystem(elements=[QuoteElement('quote')], interactions=[search], time=DiscreteTime())
        >>> simulate(system, until=3).values('quote.x').tolist()
        [1.0, 1.5, 2.0, 2.0]
    """

    def __init__(
        self,
        label: str,
        *,
        target: str,
        state: str,
        signal: Callable[..., object],
        step: float | Callable[..., object] = 0.1,
        stepping: str = 'multiplicative',
        lower: float | Callable[..., object] | None = None,
        upper: float | Callable[..., object] | None = None,
        form: str = 'symbolic',
        phase: str | None = None,
        schedule: str = 'synchronous',
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            function=self._get_next,
            form=form,
            phase=phase,
            schedule=schedule,
        )
        self.signal = signal
        self.step = step
        self.stepping = stepping
        self.lower = lower
        self.upper = upper

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown stepping, a symbolic form that is not synchronous, or an uncallable signal.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The stepping, the form, or the schedule is unknown, the symbolic form is not synchronous, or the signal
                is not callable.
        """
        super()._prepare(context)
        if self.stepping not in STEPPINGS:
            error_msg = f'The stepping of {self.label!r} is {self.stepping!r}. Use one of {", ".join(STEPPINGS)}.'
            raise ValidationError(error_msg)
        if self.form == 'symbolic' and self.schedule != 'synchronous':
            error_msg = f'The schedule {self.schedule!r} of {self.label!r} runs agent rules. Use `form="agent"`.'
            raise ValidationError(error_msg)
        if not callable(self.signal):
            error_msg = f'The signal of {self.label!r} must be callable, but it is {self.signal!r}.'
            raise ValidationError(error_msg)

    def _read(self, item: float | Callable[..., object] | None, *arguments: object) -> object:
        """Return a number, the value of a function at the arguments, or `None`."""
        return item(*arguments) if callable(item) else item

    def _get_number(
        self,
        item: float | Callable[..., object] | None,
        agent: AgentViewProtocol,
        system: SystemViewProtocol,
    ) -> float | None:
        """Return a number or the value of a function of an agent as a number, or `None`."""
        found = self._read(item, agent, system)
        if found is None:
            return None
        if not isinstance(found, SupportsFloat):
            error_msg = f'A function of {self.label!r} returns {found!r} for an agent. Return a number.'
            raise ValidationError(error_msg)
        return float(found)

    def _get_agent_next(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the next value of the state of an agent."""
        direction = float(np.sign(self._get_number(self.signal, agent, system) or 0.0))
        step = self._get_number(self.step, agent, system) or 0.0
        current = float(getattr(agent, self.state))
        value = current + step * direction if self.stepping == 'additive' else current * (1 + step * direction)
        lower, upper = self._get_number(self.lower, agent, system), self._get_number(self.upper, agent, system)
        if upper is not None:
            value = min(upper, value)
        if lower is not None:
            value = max(lower, value)
        return value

    def _get_next(self, *arguments: object) -> object:
        """Return the next value of the state, of the namespace or of an agent and its system."""
        if self.form == 'agent':
            agent, system = arguments
            return self._get_agent_next(cast(AgentViewProtocol, agent), cast(SystemViewProtocol, system))
        (namespace,) = arguments
        value = SR(getattr(self._get_member(cast(SimpleNamespace, namespace), 'target'), self.state))
        direction = sign(self._read(self.signal, namespace))
        step = SR(self._read(self.step, namespace))
        stepped = value + step * direction if self.stepping == 'additive' else value * (1 + step * direction)
        return clip(stepped, self._read(self.lower, namespace), self._read(self.upper, namespace))


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
        return -(shop.ask + shopper.travel * shopper.get_distance_to(shop))


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

    def _build_system(self) -> CompositeSystem:
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
            PriceAdjustmentInteraction(
                'settlement_price',
                target='shops',
                state='price',
                signal=lambda agent, system: agent.direction,
                step=lambda agent, system: agent.adjustment,
                lower=lambda agent, system: agent.unit_cost,
                form='agent',
                phase='accounts',
            ),
            BirthDeathInteraction(
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
            rule=PartnerMatchingSpec(choice='score', score=ResidentElement.get_preference, value='provider'),
            reserve=('bid', 'ask'),
            quantity=('demand', 'supply'),
            transfers=(('money', -1),),
            phase='market',
            on_match=StoreElement.record_sale,
        )
        return CompositeSystem(
            self.label,
            elements=[shoppers, stores],
            interactions=[*plan, shopping, *settlement],
            space=LatticeSpace((self.width, self.height), boundary='fixed', metric='chebyshev'),
            time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('market'), PhaseSpec('accounts')])),
        )
