"""Markets models: traders who wander a grid and barter two or three goods with their neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import MovementInteraction, TransferInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import ObservableSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model
from ._exchange import cobb_douglas, trade_goods

GOODS = ('sugar', 'spice', 'salt')
WEIGHTS = tuple(f'{good}_weight' for good in GOODS)


def _utility(agent: object) -> float:
    """Return the logarithm of the Cobb-Douglas utility of a trader."""
    return cobb_douglas(agent, GOODS, WEIGHTS)


def _goods_observables(system: System) -> list[ObservableSpec]:
    """Return the mean utility of the traders and the total of every good."""
    del system
    return [
        ObservableSpec('utility', _utility, group='traders', reduce='mean'),
        *(ObservableSpec(good, group='traders', variable=good, reduce='sum') for good in GOODS),
    ]


class GoodsTraderElement(Element):
    """Trader who holds goods and values them with Cobb-Douglas weights.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sugar:
            The holding of sugar.

        spice:
            The holding of spice.

        salt:
            The holding of salt.

        sugar_weight:
            The Cobb-Douglas weight of sugar.

        spice_weight:
            The Cobb-Douglas weight of spice.

        salt_weight:
            The Cobb-Douglas weight of salt, zero when salt is not traded.

    Examples:
        >>> from skcomplex.models.economics.markets import GoodsTraderElement
        >>> GroupElement('traders', element=GoodsTraderElement, size=4, values={'sugar': 5.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'sugar_weight', 'color_range': (0.0, 1.0)}),
        *,
        sugar: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        spice: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        salt: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        sugar_weight: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        spice_weight: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        salt_weight: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sugar = sugar
        self.spice = spice
        self.salt = salt
        self.sugar_weight = sugar_weight
        self.spice_weight = spice_weight
        self.salt_weight = salt_weight


class ExchangeInteraction(TransferInteraction):
    """Bilateral barter between paired traders, a transfer whose trades repeat while both traders gain.

    Every step the transfer pairs the traders, and every pair barters small quantities of every pair of goods at the
    geometric mean of their marginal rates of substitution while both gain. A trader holds every good in the state of
    its name and its Cobb-Douglas weight in the state `<good>_weight`. Every trade moves goods between the two traders,
    so the total of every good stays the same, which the default run checks as invariants. The exchange reports the
    mean `price` of the second good in units of the first, the standard deviation `dispersion` of its logarithm over
    the trades of the step, and the number of trades `volume`.

    Args:
        label:
            The label of the exchange.

        source:
            The path of the group of traders.

        target:
            The path of the group of traders, the source again.

        goods:
            The names of the traded goods.

        pairing:
            How the transfer pairs the traders, such as `'neighbor-pairs'` for disjoint pairs of neighbours or
            `'pairs'` for disjoint random pairs.

        quantity:
            The quantity of the bought good in every trade.

        phase:
            The phase the trades run in.

    Examples:
        >>> from skcomplex.models.economics.markets import ExchangeInteraction
        >>> ExchangeInteraction('exchange', source='traders', target='traders').reports
        ('price', 'dispersion', 'volume')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'dispersion', 'volume')

    def __init__(
        self,
        label: str,
        *,
        source: str,
        target: str,
        goods: tuple[str, ...] = GOODS,
        pairing: str = 'pairs',
        quantity: float = 0.1,
        phase: str = 'trade',
    ) -> None:
        super().__init__(
            label,
            source=source,
            target=target,
            state=goods[0],
            amount=quantity,
            pairing=pairing,
            form='agent',
            phase=phase,
        )
        self.goods = goods
        self.quantity = quantity

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the barter of every pair of traders and the report of the trades of the step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [*super()._get_laws(s), RuleLaw(self.report_trades, phase=self.phase)]

    def transfer(self, source: AgentViewProtocol, target: AgentViewProtocol, model: ModelViewProtocol) -> None:
        """Barter between two traders while both gain, and keep the trades of the step for its report.

        Args:
            source:
                The first trader.

            target:
                The second trader.

            model:
                The model view of the exchange.
        """
        weights = [f'{good}_weight' for good in self.goods]
        trades = trade_goods(source, target, self.goods, weights, quantity=self.quantity)
        logs = model.storage.setdefault('trades', [])
        if isinstance(logs, list):
            logs.extend(trades)

    def report_trades(self, model: ModelViewProtocol) -> None:
        """Report the mean price of the second good in the first, its dispersion, and the trades of the step.

        Args:
            model:
                The model view of the exchange.
        """
        trades = model.storage.pop('trades', [])
        trades = trades if isinstance(trades, list) else []
        logs = [math.log(price) for bought, paid, price in trades if (bought, paid) == (0, 1)]
        model.report('price', math.exp(float(np.mean(logs))) if logs else math.nan)
        model.report('dispersion', float(np.std(logs)) if len(logs) > 1 else math.nan)
        model.report('volume', len(trades))


@register_model(
    field='economics/markets',
    tags=('market', 'agents', 'barter', 'price convergence', 'grid'),
    references=('Gooding 2019 ch. 8-12', 'Epstein and Axtell 1996'),
    defaults={'until': 50, 'invariants': dict.fromkeys(GOODS, 1e-6)},
    observables=_goods_observables,
)
def goods_traders(
    n_traders: int = 200,
    *,
    goods: int = 3,
    width: int = 20,
    height: int = 20,
    quantity: float = 0.1,
    mobility: float = 0.5,
    low: float = 1.0,
    high: float = 20.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return traders who wander a grid and barter two or three goods with their neighbours.

    Every trader holds random amounts of the goods and values them with random Cobb-Douglas weights. Every step the
    traders wander to a random neighbouring cell, then every trader is paired with a neighbour, and the pairs barter
    while both gain, at the geometric mean of their marginal rates of substitution. Trades only move goods between the
    traders, so the total of every good is unchanged, while the mean utility rises and the prices of different trades
    converge, so the dispersion of the logarithm of the price falls, the toy and three-goods traders of Gooding (2019).

    Args:
        n_traders:
            The number of traders.

        goods:
            The number of goods, 2 for sugar and spice or 3 with salt.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        quantity:
            The quantity of the bought good in every trade.

        mobility:
            The probability that a trader moves in a step.

        low:
            The smallest initial holding of a good.

        high:
            The largest initial holding of a good.

        random_state:
            The seed or generator of the cells, holdings, and weights.

    Returns:
        system:
            The model, with the group `traders`, their random walk `wander`, and the exchange `exchange`, whose
            reporters are `exchange.price`, `exchange.dispersion`, and `exchange.volume`.

    Raises:
        ValueError:
            The number of goods is not 2 or 3.

    Examples:
        >>> from skcomplex.models.economics.markets import goods_traders
        >>> goods_traders(20, goods=2).elements[0].check().prepare().size_
        20
    """
    if goods not in (2, 3):
        error_msg = f'The traders barter 2 or 3 goods, but got {goods}.'
        raise ValueError(error_msg)
    rng = np.random.default_rng(random_state)
    cells = [(x, y) for x in range(width) for y in range(height)]
    locations = [cells[index] for index in rng.integers(len(cells), size=n_traders)]
    weights = rng.dirichlet(np.ones(goods), size=n_traders)
    holdings = rng.uniform(low, high, (n_traders, goods))
    values: dict[str, Array] = {GOODS[index]: holdings[:, index] for index in range(goods)}
    values |= {WEIGHTS[index]: weights[:, index] for index in range(goods)}
    traders = GroupElement(
        'traders',
        element=GoodsTraderElement,
        size=n_traders,
        values=values,
        variable=True,
        location=locations,
    )
    return System(
        elements=[traders],
        interactions=[
            MovementInteraction('wander', target='traders', probability=mobility, phase='move'),
            ExchangeInteraction(
                'exchange',
                source='traders',
                target='traders',
                goods=GOODS[:goods],
                pairing='neighbor-pairs',
                quantity=quantity,
            ),
        ],
        space=LatticeSpace((width, height), boundary='periodic', neighborhood='moore'),
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('move'), PhaseSpec('trade')])),
    )
