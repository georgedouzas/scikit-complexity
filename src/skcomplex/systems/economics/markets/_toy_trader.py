"""Markets systems: Gooding's Toy Trader.

Traders with money and toys buy the cheapest toy they can see.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.policy import PolicyRuleInteraction

FITNESS_TESTS = ('none', 'money', 'assets')
GOODS = ('toys', 'crayons', 'blackboards')
STOCKS = (5.0, 30.0, 15.0)
TARGETS = ((3.0, 3.0), (10.0, 15.0), (5.0, 8.0))
PRICE_UP, PRICE_DOWN, LOWEST_PRICE = 1.0101, 0.99, 0.5
COPY = 0.5


class ToyTraderView(AgentView):
    """Trader as the market sees it: its place, money, goods, prices, and pricing rule."""

    x: float
    y: float
    money: float
    toys: float
    crayons: float
    blackboards: float
    toy_price: float
    crayon_price: float
    blackboard_price: float
    price_up: float
    price_down: float


class ToyTraderElement(Element):
    """Trader on a torus with money, toys, crayons, and blackboards.

    It holds its selling prices and its pricing rule.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The horizontal place of the trader.

        y:
            The vertical place of the trader.

        money:
            The money of the trader.

        toys:
            The toys of the trader.

        crayons:
            The crayons of the trader.

        blackboards:
            The blackboards of the trader.

        toy_price:
            The price the trader sells a toy at.

        crayon_price:
            The price the trader sells a crayon at.

        blackboard_price:
            The price the trader sells a blackboard at.

        price_up:
            The factor that raises the toy price when the toys run short.

        price_down:
            The factor that lowers the toy price when the toys pile up.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import ToyTraderElement
        >>> GroupElement('traders', element=ToyTraderElement, size=3).size
        3
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'money'}),
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        money: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        toys: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        crayons: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        blackboards: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        toy_price: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 10.0,
        crayon_price: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 10.0,
        blackboard_price: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 10.0,
        price_up: Annotated[float, StateSpec(domain=RealSet.closed_open(1, oo))] = PRICE_UP,
        price_down: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = PRICE_DOWN,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.money = money
        self.toys = toys
        self.crayons = crayons
        self.blackboards = blackboards
        self.toy_price = toy_price
        self.crayon_price = crayon_price
        self.blackboard_price = blackboard_price
        self.price_up = price_up
        self.price_down = price_down

    @classmethod
    def _get_view(cls) -> type[ToyTraderView]:
        """Return the view of the agents of the element."""
        return ToyTraderView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the market moves and trades for every trader.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class ToyMarketInteraction(PolicyRuleInteraction[SystemView]):
    """Market of the Toy Trader.

    Every trader buys the cheapest good it can see, sellers price by inventory, and all move. Every iteration the
    traders, in random order, try to buy one unit of every good from the trader with the lowest price within the radius
    `sight`, picking at random among equal prices, and buy it when they can pay. After a sale the seller lowers its
    price by its factor `price_down` when its stock is above the target and raises it by `price_up` when it is below,
    never below 0.5. Then every trader moves one unit in a random direction on the torus. Toys are priced by the
    trader's own factors, which mutate with the probability `mutation` by the difference of two uniform draws on
    `[0, 0.01)`, and crayons and blackboards by the factors 1.0101 and 0.99. With a fitness test, a trader of the
    poorest half copies with the probability `imitation` the toy price of a random trader of the richest `elite` share
    and, each with probability one half, its factors plus a uniform mutation on `[-0.01, 0.01]`.

    Args:
        label:
            The label of the market.

        rule:
            Unused, since the market overrides `act`.

        phase:
            The phase the market acts in, or `None` for the default phase.

        traders:
            The path of the group of traders.

        sight:
            The radius within which a trader sees prices, its market sight.

        mutation:
            The probability per iteration that a factor of the toy price mutates.

        imitation:
            The probability per iteration that a poor trader imitates a rich one.

        elite:
            The share of the richest traders that poor traders imitate.

        size:
            The side of the torus.

        n_goods:
            The goods traded, 1 for toys only or 3 for toys, crayons, and blackboards.

        fitness:
            The fitness test of imitation, `'none'`, `'money'`, or `'assets'`.

    Raises:
        ValidationError:
            The fitness test or the number of goods is unknown, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.economics.markets import ToyMarketInteraction
        >>> ToyMarketInteraction('market', traders='traders').reports[:2]
        ('average_price', 'trades')
    """

    draws = True

    reports: ClassVar[tuple[str, ...]] = ('average_price', 'trades', 'money_gini', 'asset_gini')

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        traders: Annotated[str, RoleSpec()],
        sight: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
        imitation: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        elite: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.05,
        size: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 51.0,
        n_goods: int = 1,
        fitness: str = 'none',
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.traders = traders
        self.sight = sight
        self.mutation = mutation
        self.imitation = imitation
        self.elite = elite
        self.size = size
        self.n_goods = n_goods
        self.fitness = fitness

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown fitness test or number of goods.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The fitness test or the number of goods is unknown.
        """
        super()._prepare(context)
        if self.fitness not in FITNESS_TESTS:
            error_msg = f'The fitness test of {self.label!r} is {self.fitness!r}. Use one of {FITNESS_TESTS}.'
            raise ValidationError(error_msg)
        if self.n_goods not in {1, len(GOODS)}:
            error_msg = f'{self.label!r} trades {self.n_goods} goods. Use 1 or {len(GOODS)}.'
            raise ValidationError(error_msg)

    @staticmethod
    def _get_gini(values: Array) -> float:
        """Return the Gini coefficient of nonnegative values."""
        ordered = np.sort(values)
        total = ordered.sum()
        if total <= 0:
            return 0.0
        ranks = 2 * np.arange(1, len(ordered) + 1) - len(ordered) - 1
        return float(ranks @ ordered / (len(ordered) * total))

    def _reprice(self, state: dict[str, Array], seller: int, good: int) -> None:
        """Lower the price of a good of a seller above its target stock and raise it below."""
        low, high = TARGETS[good]
        stock = state['stocks'][seller, good]
        up, down = (state['up'][seller], state['down'][seller]) if good == 0 else (PRICE_UP, PRICE_DOWN)
        if stock > high:
            state['prices'][seller, good] = max(state['prices'][seller, good] * down, LOWEST_PRICE)
        elif stock < low:
            state['prices'][seller, good] *= up

    def _trade(self, system: SystemView, state: dict[str, Array]) -> list[float]:
        """Let every trader buy one unit of every good from the cheapest trader in sight.

        The traders buy in random order.
        """
        rng, sight, size = system.rng, system.get_parameter('sight'), system.get_parameter('size')
        places, sold = state['places'], []
        for buyer in rng.permutation(len(places)):
            gap = np.abs(places - places[buyer])
            gap = np.minimum(gap, size - gap)
            near = np.hypot(gap[:, 0], gap[:, 1]) <= sight
            near[buyer] = False
            for good in range(self.n_goods):
                sellers = np.flatnonzero(near & (state['stocks'][:, good] > 0))
                if not len(sellers):
                    continue
                prices = state['prices'][sellers, good]
                cheapest = sellers[prices == prices.min()]
                seller = int(cheapest[rng.integers(len(cheapest))])
                price = state['prices'][seller, good]
                if state['money'][buyer] < price:
                    continue
                state['money'][buyer] -= price
                state['money'][seller] += price
                state['stocks'][buyer, good] += 1
                state['stocks'][seller, good] -= 1
                if good == 0:
                    sold.append(price)
                self._reprice(state, seller, good)
        return sold

    def _evolve(self, system: SystemView, state: dict[str, Array], average: float) -> None:
        """Mutate the pricing factors and let poor traders imitate rich ones.

        The imitation passes the fitness test.
        """
        rng, count = system.rng, len(state['money'])
        mutation = system.get_parameter('mutation')
        for name, low, high in (('up', 1.0, np.inf), ('down', 0.001, 1.0)):
            mutating = rng.random(count) < mutation
            shift = rng.uniform(0, 0.01, count) - rng.uniform(0, 0.01, count)
            state[name] = np.where(mutating, np.clip(state[name] + shift, low, high), state[name])
        if self.fitness == 'none':
            return
        wealth = state['money'] + (state['stocks'][:, 0] * average if self.fitness == 'assets' else 0.0)
        order = np.argsort(wealth)
        poor, rich = order[: count // 2], order[-max(1, int(system.get_parameter('elite') * count)) :]
        for trader in poor[rng.random(len(poor)) < system.get_parameter('imitation')]:
            model = int(rich[rng.integers(len(rich))])
            state['prices'][trader, 0] = state['prices'][model, 0]
            if rng.random() < COPY:
                state['up'][trader] = max(state['up'][model] + rng.uniform(-0.01, 0.01), 1.0)
            if rng.random() < COPY:
                state['down'][trader] = float(np.clip(state['down'][model] + rng.uniform(-0.01, 0.01), 0.001, 1.0))

    def act(self, system: SystemView) -> None:
        """Run one iteration: trade, evolve the pricing rules, and move.

        Args:
            system:
                The system view of the market.
        """
        traders = sorted(system.get_agents('traders', ToyTraderView), key=lambda trader: trader.id)
        prices_of = ('toy_price', 'crayon_price', 'blackboard_price')
        state = {
            'places': np.array([[trader.x, trader.y] for trader in traders]),
            'money': np.array([trader.money for trader in traders]),
            'stocks': np.array([[getattr(trader, good) for good in GOODS] for trader in traders]),
            'prices': np.array([[getattr(trader, name) for name in prices_of] for trader in traders]),
            'up': np.array([trader.price_up for trader in traders]),
            'down': np.array([trader.price_down for trader in traders]),
        }
        sold = self._trade(system, state)
        last = system.get_history('average_price', 1)
        fallback = float(last[-1]) if len(last) else float(state['prices'][:, 0].mean())
        average = float(np.mean(sold)) if sold else fallback
        self._evolve(system, state, average)
        angles = system.rng.uniform(0, 2 * np.pi, len(traders))
        steps = np.column_stack((np.cos(angles), np.sin(angles)))
        state['places'] = (state['places'] + steps) % system.get_parameter('size')
        for index, trader in enumerate(traders):
            trader.x, trader.y = (float(value) for value in state['places'][index])
            trader.money = float(state['money'][index])
            for good, name, value in zip(GOODS, prices_of, state['prices'][index], strict=True):
                setattr(trader, good, float(state['stocks'][index, GOODS.index(good)]))
                setattr(trader, name, float(value))
            trader.price_up, trader.price_down = float(state['up'][index]), float(state['down'][index])
        system.report('average_price', average)
        system.report('trades', float(len(sold)))
        system.report('money_gini', self._get_gini(state['money']))
        system.report('asset_gini', self._get_gini(state['money'] + state['stocks'][:, 0] * average))


class ToyTraderSystem(BaseSystem):
    """Gooding's Toy Trader.

    Traders with money and toys buy the cheapest toy they can see. Four hundred traders on a 51 by 51 torus start with
    100 money, 5 toys, and a toy price of 10. Every iteration every trader buys one toy from the cheapest trader within
    its market sight, if it can pay, and moves one unit in a random direction. A seller keeps its stock at 3 toys by
    lowering its price by the factor 0.99 above it and raising it by 1.0101 below it (Gooding 2019, Section 8.2 and the
    appendix). Money and toys are conserved and supply exceeds the demand for stocks, yet the price depends on the
    market sight: it falls when traders see only their neighbours and rises when they see everyone, so supply and demand
    alone do not set it (Section 10.1). Mutating pricing factors make the traders heterogeneous and raise prices and
    inequality, and the imitation of the richest 5 percent by the poorest half under a fitness test of money or of
    assets evolves the price patterns of speculation and asset markets (Sections 10.2 and 11). With `n_goods=3` the
    traders also trade crayons and blackboards, starting with 30 and 15, which keep the fixed factors and the target
    stocks of 10 to 15 and 5 to 8 of the Three-Goods Trader (Section 11.1).

    The system has the group `traders` and the market `market`, whose reporters are the average price of the toys traded
    `market.average_price`, the toys traded `market.trades`, and the Gini coefficients of money `market.money_gini` and
    of assets `market.asset_gini`.

    Args:
        n_traders:
            The number of traders.

        sight:
            The radius within which a trader sees prices.

        mutation:
            The probability per iteration that a factor of the toy price mutates, such as 1 / 5000.

        imitation:
            The probability per iteration that a poor trader imitates a rich one under a fitness test.

        fitness:
            The fitness test of imitation, `'none'`, `'money'`, or `'assets'`.

        n_goods:
            The goods traded, 1 for toys only or 3 for toys, crayons, and blackboards.

        size:
            The side of the torus.

        random_state:
            The seed or generator of the places of the traders.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import ToyTraderSystem
        >>> ToyTraderSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'economics/markets'
    topics = ('money', 'prices', 'agents', 'inequality', 'evolution', 'imitation', 'market sight')
    references = ('Gooding 2019',)
    until = 300

    def __init__(
        self,
        n_traders: int = 400,
        *,
        sight: float = 1.0,
        mutation: float = 0.0,
        imitation: float = 0.1,
        fitness: str = 'none',
        n_goods: int = 1,
        size: float = 51.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_traders = n_traders
        self.sight = sight
        self.mutation = mutation
        self.imitation = imitation
        self.fitness = fitness
        self.n_goods = n_goods
        self.size = size
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        places = rng.uniform(0, self.size, (self.n_traders, 2))
        stocks = dict(zip(GOODS, STOCKS[: self.n_goods] + (0.0,) * (len(GOODS) - self.n_goods), strict=True))
        traders = GroupElement(
            'traders',
            element=ToyTraderElement,
            size=self.n_traders,
            values={'x': places[:, 0], 'y': places[:, 1], **stocks},
        )
        market = ToyMarketInteraction(
            'market',
            traders='traders',
            sight=self.sight,
            mutation=self.mutation,
            imitation=self.imitation,
            size=self.size,
            n_goods=self.n_goods,
            fitness=self.fitness,
        )
        return CompositeSystem(self.label, elements=[traders], interactions=[market])
