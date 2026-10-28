"""Markets models: traders who wander a grid and barter two or three goods with their neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, Any, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, MatchLaw, RuleLaw
from ....core.laws.utils import matches
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.specs import ObservableSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import AgentView, ModelView
from ..._catalog import register_model
from ._exchange import cobb_douglas, neighbor_pairing, trade_goods

type Pairing = Callable[[Sequence[Any], Sequence[Any], np.random.Generator, int], list[tuple[Any, Any]]]

GOODS = ('sugar', 'spice', 'salt')
WEIGHTS = tuple(f'{good}_weight' for good in GOODS)


def _utility(agent: Any) -> float:  # noqa: ANN401
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
        mobility: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sugar = sugar
        self.spice = spice
        self.salt = salt
        self.sugar_weight = sugar_weight
        self.spice_weight = spice_weight
        self.salt_weight = salt_weight
        self.mobility = mobility

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the wandering rule of the phase of moves.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.wander, phase='move')]

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


class ExchangeInteraction(Interaction):
    """Bilateral barter between traders, paired every step by the matching law.

    Every pair barters small quantities of every pair of goods at the geometric mean of their marginal rates of
    substitution while both gain. A trader holds every good in the state of its name and its Cobb-Douglas weight in
    the state `<good>_weight`. Every trade is a transfer between the two traders, so the total of every good stays
    the same, which the default run checks as invariants. The exchange reports the mean `price` of the second good in
    units of the first, the standard deviation `dispersion` of its logarithm over the trades of the step, and the
    number of trades `volume`.

    Args:
        label:
            The label of the exchange.

        goods:
            The names of the traded goods.

        pairing:
            The pairing of the matching law, such as `neighbor_pairing`, or `None` for random pairs.

        quantity:
            The quantity of the bought good in every trade.

        **values:
            The values of the declared variables, by name:

            traders:
                The path of the group of traders.

    Examples:
        >>> from skcomplex.models.economics.markets import ExchangeInteraction
        >>> ExchangeInteraction('exchange', traders='traders').reports
        ('price', 'dispersion', 'volume')
    """

    reports: ClassVar[tuple[str, ...]] = ('price', 'dispersion', 'volume')

    def __init__(
        self,
        label: str,
        *,
        traders: Annotated[str, RoleSpec()],
        goods: tuple[str, ...] = GOODS,
        pairing: Pairing | None = None,
        quantity: float = 0.1,
    ) -> None:
        super().__init__(label)
        self.traders = traders
        self.goods = goods
        self.pairing = pairing
        self.quantity = quantity

    def _get_laws(self, s: SimpleNamespace) -> list[Any]:
        """Return the matching of neighbours and the barter rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            MatchLaw('traders', pairing=self.pairing, label='partners', phase='trade'),
            RuleLaw(self.trade, phase='trade'),
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
            The model, with the group `traders` and the exchange `exchange`, whose reporters are `exchange.price`,
            `exchange.dispersion`, and `exchange.volume`.

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
    values: dict[str, Any] = {GOODS[index]: holdings[:, index] for index in range(goods)}
    values |= {WEIGHTS[index]: weights[:, index] for index in range(goods)}
    traders = GroupElement(
        'traders',
        element=GoodsTraderElement,
        size=n_traders,
        values={'mobility': mobility, **values},
        variable=True,
        location=locations,
    )
    return System(
        elements=[traders],
        interactions=[
            ExchangeInteraction(
                'exchange',
                traders='traders',
                goods=GOODS[:goods],
                pairing=neighbor_pairing,
                quantity=quantity,
            ),
        ],
        space=LatticeSpace((width, height), boundary='periodic', neighborhood='moore'),
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('move'), PhaseSpec('trade')])),
    )
