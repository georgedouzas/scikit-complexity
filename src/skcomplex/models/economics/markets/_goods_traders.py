"""Markets models: traders who wander a grid and barter two or three goods with their neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np

from ....core.backends import AgentView, ModelView
from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction, Match, matches
from ....core.laws import Invariant, Law, Rule, Symbols
from ....core.runtime import register_model
from ....core.space import DiscreteTime, LatticeSpace, Phase, Phases
from ....core.system import Observable, System
from ..realizations import NeighbourPairing, cobb_douglas, trade_goods

type Pairing = Callable[[Sequence[Any], Sequence[Any], np.random.Generator, int], list[tuple[Any, Any]]]

GOODS = ('sugar', 'spice', 'salt')
WEIGHTS = tuple(f'{good}_weight' for good in GOODS)


def _total(good: str, model: Any) -> float:  # noqa: ANN401
    """Return the total holding of a good over the traders bound to an exchange."""
    return float(sum(getattr(agent, good) for agent in model.agents('traders')))


def _utility(agent: Any) -> float:  # noqa: ANN401
    """Return the logarithm of the Cobb-Douglas utility of a trader."""
    return cobb_douglas(agent, GOODS, WEIGHTS)


def _goods_observables(system: System) -> list[Observable]:
    """Return the mean utility of the traders and the total of every good."""
    del system
    return [
        Observable('utility', _utility, group='traders', reduce='mean'),
        *(Observable(good, group='traders', variable=good, reduce='sum') for good in GOODS),
    ]


class GoodsTrader(Element):
    """Trader who holds goods, values them with Cobb-Douglas weights, and wanders to a random neighbouring cell.

    Args:
        label:
            The label of the agent or group.

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

        mobility:
            The probability that the trader moves in a step.

    Examples:
        >>> from skcomplex.models.economics.markets import GoodsTrader
        >>> GoodsTrader.group('traders', 4, sugar=5.0).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'sugar_weight', 'color_range': (0.0, 1.0)})
    sugar: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    spice: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    salt: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    sugar_weight: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    spice_weight: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    salt_weight: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    mobility: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the wandering rule of the phase of moves.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.wander, phase='move')]

    def wander(self, agent: AgentView, model: ModelView) -> None:
        """Move to a random neighbouring cell with the probability of the mobility.

        Args:
            agent:
                The trader.

            model:
                The model.
        """
        if model.rng.random() >= model.parameter('mobility'):
            return
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(model.rng.integers(len(cells)))])


class Exchange(Interaction):
    """Bilateral barter between traders, paired every step by the matching law.

    Every pair barters small quantities of every pair of goods at the geometric mean of their marginal rates of
    substitution while both gain. A trader holds every good in the state of its name and its Cobb-Douglas weight in
    the state `<good>_weight`. Every trade is a transfer between the two traders, and the invariants of the exchange
    check that the total of every good stays the same. The exchange reports the mean `price` of the second good in
    units of the first, the standard deviation `dispersion` of its logarithm over the trades of the step, and the
    number of trades `volume`.

    Args:
        label:
            The label of the exchange.

        traders:
            The path of the group of traders.

        goods:
            The names of the traded goods.

        pairing:
            The pairing of the matching law, such as `NeighbourPairing()`, or `None` for random pairs.

        quantity:
            The quantity of the bought good in every trade.

    Examples:
        >>> from skcomplex.models.economics.markets import Exchange
        >>> Exchange('exchange', traders='traders').reports
        ('price', 'dispersion', 'volume')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'dispersion', 'volume')
    traders: str = role()
    goods: tuple[str, ...] = GOODS
    pairing: Pairing | None = None
    quantity: float = 0.1

    def laws(self, s: Symbols) -> list[Any]:
        """Return the matching of neighbours, the barter rule, and the invariant of the total of every good.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Match('traders', pairing=self.pairing, label='partners', phase='trade'),
            Rule(self.trade, phase='trade'),
            *(Invariant(f'total {good}', functools.partial(_total, good), tolerance=1e-6) for good in self.goods),
        ]

    def trade(self, model: Any) -> None:  # noqa: ANN401
        """Barter between every pair of partners and report the prices of the second good in the first.

        Args:
            model:
                The model view of the exchange.
        """
        weights = [f'{good}_weight' for good in self.goods]
        logs, volume = [], 0
        for first, second in matches(model, 'partners'):
            trades = trade_goods(first, second, self.goods, weights, quantity=self.quantity)
            volume += len(trades)
            logs += [math.log(price) for bought, paid, price in trades if (bought, paid) == (0, 1)]
        model.report('price', math.exp(float(np.mean(logs))) if logs else math.nan)
        model.report('dispersion', float(np.std(logs)) if len(logs) > 1 else math.nan)
        model.report('volume', volume)


@register_model(
    field='economics/markets',
    tags=('market', 'agents', 'barter', 'price convergence', 'grid'),
    references=('Gooding 2019 ch. 8-12', 'Epstein and Axtell 1996'),
    defaults={'until': 50},
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
            The model, with the group `traders` and the exchange `exchange`, whose reporters are `exchange.price`,
            `exchange.dispersion`, and `exchange.volume`.

    Raises:
        ValueError:
            The number of goods is not 2 or 3.

    Examples:
        >>> from skcomplex.models.economics.markets import goods_traders
        >>> goods_traders(20, goods=2).elements[0].size
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
    values = {GOODS[index]: holdings[:, index] for index in range(goods)}
    values |= {WEIGHTS[index]: weights[:, index] for index in range(goods)}
    traders = GoodsTrader.group('traders', n_traders, variable=True, location=locations, mobility=mobility, **values)
    return System(
        components=[traders],
        interactions=[
            Exchange(
                'exchange',
                traders='traders',
                goods=GOODS[:goods],
                pairing=NeighbourPairing(),
                quantity=quantity,
            ),
        ],
        space=LatticeSpace((width, height), boundary='periodic', neighborhood='moore'),
        time=DiscreteTime(schedule=Phases([Phase('move'), Phase('trade')])),
    )
