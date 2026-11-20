"""Markets systems: shops on a grid that compete for nearby shoppers.

It also holds the digital market that removes distance.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, SupportsFloat

import numpy as np
from sage.all__sagemath_symbolics import SR, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import clip, sign
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import AgentView, BaseSolver, MesaSolver, SystemView
from ....core.errors import ValidationError
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.agents import BirthDeathInteraction
from ...general.matching import MatchingInteraction, MatchSpec, PartnerMatchingSpec

STEPPINGS = ('multiplicative', 'additive')
SCHEDULES = (None, 'synchronous', 'sequential', 'random-sequential', 'member')


class PriceAdjustmentInteraction(Interaction):
    """Price adjustment of a state of discrete time by the sign of a signal, kept within bounds.

    Economics names it a price adjustment by trial and error, as a shop that raises its price while demand exceeds its
    supply and lowers it otherwise. With the stepping `'multiplicative'` the next value of the state `x` is
    `x (1 + step sign(signal))`, and with the stepping `'additive'` it is `x + step sign(signal)`, so the state rises
    while the signal is positive, falls while it is negative, and stays while it is zero, by trial and error rather
    than by the size of the signal. The next value is then limited to `[lower, upper]`.

    Without a `schedule` it sets an update law, and the signal, the step, and the bounds are numbers or functions of the
    namespace, where `namespace.target` holds the states and parameters of the target. With a `schedule` it sets one
    rule that walks the agents of the target in that order, and they are numbers or functions `f(agent, system)` that
    return numbers. Under the schedule `'synchronous'` every agent steps from the values at the start of the rule,
    under `'sequential'` or `'random-sequential'` one after the other, in their order or in a new random order, and
    under `'member'` each in its own activation.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

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

        phase:
            The phase of the step the rule of a schedule runs in, or `None` for the default phase.

        schedule:
            The order in which a rule walks the agents, `'synchronous'`, `'sequential'`, `'random-sequential'`, or
            `'member'`, or `None` for an update law of the state of the target.

    Raises:
        ValidationError:
            The stepping or the schedule is unknown, or the signal is not callable, raised by `prepare`.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.economics.markets import PriceAdjustmentInteraction
        >>> search = PriceAdjustmentInteraction(
        ...     'search', target='quote', state='x', signal=lambda s: 2 - s.target.x, step=0.5, upper=2.0
        ... )
        >>> system = System(elements=[Element('quote', values={'x': 1.0})], interactions=[search])
        >>> system.prepare().simulate(until=3).get_values('quote.x').tolist()
        [1.0, 1.5, 2.0, 2.0]
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str,
        signal: Callable[..., object],
        step: float | Callable[..., object] = 0.1,
        stepping: str = 'multiplicative',
        lower: float | Callable[..., object] | None = None,
        upper: float | Callable[..., object] | None = None,
        phase: str | None = None,
        schedule: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.signal = signal
        self.step = step
        self.stepping = stepping
        self.lower = lower
        self.upper = upper
        self.phase = phase
        self.schedule = schedule

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the step of the state, or the rule of the agents, as the laws.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the target.

        Raises:
            ValidationError:
                The stepping or the schedule is unknown, or the signal is not callable.
        """
        super()._prepare(context)
        if self.schedule not in SCHEDULES:
            error_msg = f'The schedule of {self.label!r} is {self.schedule!r}. Use one of {SCHEDULES}.'
            raise ValidationError(error_msg)
        if self.stepping not in STEPPINGS:
            error_msg = f'The stepping of {self.label!r} is {self.stepping!r}. Use one of {", ".join(STEPPINGS)}.'
            raise ValidationError(error_msg)
        if not callable(self.signal):
            error_msg = f'The signal of {self.label!r} must be callable, but it is {self.signal!r}.'
            raise ValidationError(error_msg)
        if not isinstance(context, SimpleNamespace):
            return
        if self.schedule == 'member':
            self.laws_: list[BaseComponent] = [UpdateLaw(context.target, self._step_member, phase=self.phase)]
        elif self.schedule is not None:
            self.laws_ = [UpdateLaw(context, self.step_agents, phase=self.phase)]
        else:
            self.laws_ = [UpdateLaw(getattr(context.target, self.state), self._get_next(context))]

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

    def _step_member(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Step the state of one agent, in its own activation."""
        if agent.alive:
            setattr(agent, self.state, self._get_agent_next(agent, system))

    def step_agents(self, system: SystemViewProtocol) -> None:
        """Step the state of every agent by the schedule.

        Args:
            system:
                The system view of the adjustment.
        """
        agents = list(system.get_agents('target'))
        if self.schedule == 'synchronous':
            values = [self._get_agent_next(agent, system) for agent in agents]
            for agent, value in zip(agents, values, strict=True):
                setattr(agent, self.state, value)
            return
        order = system.rng.permutation(len(agents)) if self.schedule == 'random-sequential' else range(len(agents))
        for index in order:
            self._step_member(agents[int(index)], system)

    def _get_next(self, namespace: SimpleNamespace) -> object:
        """Return the next value of the state of the update law."""
        value = SR(getattr(namespace.target, self.state))
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

        values:
            Further values by name, such as the cell `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The grid whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.markets import ResidentElement
        >>> ResidentElement('shoppers', online=1.0, size=3).size
        3
    """

    visual = VisualSpec('agents', {'color_by': 'travel', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        bid: Annotated[object, RealSet.closed_open(0, oo)] = 20.0,
        demand: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        money: object = 1000.0,
        travel: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        distance_cost: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        online: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
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

        values:
            Further values by name, such as the cell `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The grid whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.markets import StoreElement
        >>> StoreElement('shops', price=10.0, size=4).size
        4
    """

    visual = VisualSpec('agents', {'color_by': 'price'})

    def __init__(
        self,
        label: str,
        *,
        price: Annotated[object, RealSet.closed_open(0, oo)] = 10.0,
        ask: Annotated[object, RealSet.closed_open(0, oo)] = 10.0,
        supply: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        sales: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        money: object = 100.0,
        profit: object = 0.0,
        direction: object = 1.0,
        capacity: Annotated[object, RealSet.open(0, oo)] = 1000.0,
        unit_cost: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        fixed_cost: Annotated[object, RealSet.closed_open(0, oo)] = 20.0,
        adjustment: Annotated[object, RealSet.closed_open(0, oo)] = 0.05,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
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


def _get_largest_share(sales: object) -> float:
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        Observable('open_shops', group='shops', variable='price', reduce='count'),
        Observable('mean_price', group='shops', variable='price', reduce='mean'),
        Observable('concentration', group='shops', variable='sales', reduce=_get_largest_share),
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_shops = n_shops
        self.width = width
        self.height = height
        self.online = online
        self.distance_cost = distance_cost
        self.capacity = capacity
        self.fixed_cost = fixed_cost
        self.value = value
        self.random_state = random_state

    @staticmethod
    def _open(system: SystemView) -> None:
        """Open the market, where every shopper wants one unit and every shop offers its capacity.

        A shopper shops online, at no travel cost, with its probability `online`, and in person otherwise.
        """
        shoppers = system.get_agents('shoppers', ResidentView)
        for shopper in shoppers:
            shopper.demand = 1.0
        for shopper in shoppers:
            shopper.travel = 0.0 if system.rng.random() < shopper.online else float(shopper.distance_cost)
        shops = system.get_agents('shops', StoreView)
        for shop in shops:
            shop.supply = float(shop.capacity)
        for shop in shops:
            shop.ask = float(shop.price)
        for shop in shops:
            shop.sales = 0.0

    @staticmethod
    def _settle(system: SystemView) -> None:
        """Settle the accounts of every shop.

        A shop turns its price back when its profit falls, records the profit, and pays its costs. The pricing then
        steps the price by the sign of the direction.
        """
        shops = system.get_agents('shops', StoreView)
        for shop in shops:
            shop.direction = float(-shop.direction if StoreElement.get_profit(shop) < shop.profit else shop.direction)
        for shop in shops:
            shop.profit = float(StoreElement.get_profit(shop))
        for shop in shops:
            shop.money = float(shop.money - shop.sales * shop.unit_cost - shop.fixed_cost)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        places = rng.choice(len(cells), size=min(self.n_shops, len(cells)), replace=False)
        space = LatticeSpace((self.width, self.height), boundary='fixed', metric='chebyshev')
        xs, ys = np.array(cells).T
        shoppers = ResidentElement(
            'shoppers',
            bid=self.value,
            online=self.online,
            distance_cost=self.distance_cost,
            values={'x': xs, 'y': ys},
            size=len(cells),
            space=space,
        )
        stores = StoreElement(
            'shops',
            price=rng.uniform(0.25 * self.value, 0.75 * self.value, len(places)),
            capacity=self.capacity,
            fixed_cost=self.fixed_cost,
            values={'x': xs[places], 'y': ys[places]},
            size=len(places),
            space=space,
        )
        plan = [
            Interaction(
                'opening',
                roles={'shoppers': 'shoppers', 'shops': 'shops'},
                laws=lambda s: [UpdateLaw(s, ShopsSystem._open, phase='market')],
            ),
        ]
        settlement = [
            Interaction(
                'settlement',
                roles={'shops': 'shops'},
                laws=lambda s: [UpdateLaw(s, ShopsSystem._settle, phase='accounts')],
            ),
            PriceAdjustmentInteraction(
                'settlement_price',
                target='shops',
                state='price',
                signal=lambda agent, system: agent.direction,
                step=lambda agent, system: agent.adjustment,
                lower=lambda agent, system: agent.unit_cost,
                schedule='synchronous',
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
        self.system_ = System(
            self.label,
            elements=[shoppers, stores],
            interactions=[*plan, shopping, *settlement],
            space=space,
            solver=MesaSolver(phases=[PhaseSpec('market'), PhaseSpec('accounts')]),
        )
