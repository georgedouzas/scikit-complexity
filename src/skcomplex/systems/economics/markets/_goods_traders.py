"""Markets systems: traders who wander a grid and barter goods.

They barter two or three goods with their neighbours.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MovementInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.exchange import ExchangeInteraction, cobb_douglas
from ..common.exchange._exchange_interaction import GOODS


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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import GoodsTraderElement
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


WEIGHTS = tuple(f'{good}_weight' for good in GOODS)


def _utility(agent: object) -> float:
    """Return the logarithm of the Cobb-Douglas utility of a trader."""
    return cobb_douglas(agent, GOODS, WEIGHTS)


class GoodsTradersSystem(BaseSystem):
    """Traders who wander a grid and barter two or three goods with their neighbours.

    Every trader holds random amounts of the goods and values them with random Cobb-Douglas weights. Every step the
    traders wander to a random neighbouring cell, then every trader is paired with a neighbour, and the pairs barter
    while both gain, at the geometric mean of their marginal rates of substitution. Trades only move goods between the
    traders, so the total of every good is unchanged, while the mean utility rises and the prices of different trades
    converge, so the dispersion of the logarithm of the price falls.

    The trade rule is the bilateral barter of sugar and spice of Epstein and Axtell (1996), and salt extends it to a
    third good. The Toy Trader of Gooding (2019), with money, posted prices, and imitation of price rules, is a
    different model. The system has the group `traders`, their random walk `wander`, and the exchange `exchange`, whose
    reporters are `exchange.price`, `exchange.dispersion`, and `exchange.volume`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValueError:
            The number of goods is not 2 or 3.

    Examples:
        >>> from skcomplex.systems.economics.markets import GoodsTradersSystem
        >>> GoodsTradersSystem(20, goods=2).prepare().system_.elements[0].prepare().size_
        20
    """

    field = 'economics/markets'
    topics = ('market', 'agents', 'barter', 'price convergence', 'grid')
    references = ('Epstein and Axtell 1996',)
    until = 50
    invariants: ClassVar[Mapping[str, float]] = dict.fromkeys(GOODS, 1e-6)
    observables = (
        ObservableSpec('utility', _utility, group='traders', reduce='mean'),
        *(ObservableSpec(good, group='traders', variable=good, reduce='sum') for good in GOODS),
    )

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_traders = n_traders
        self.goods = goods
        self.width = width
        self.height = height
        self.quantity = quantity
        self.mobility = mobility
        self.low = low
        self.high = high
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        if self.goods not in (2, 3):
            error_msg = f'The traders barter 2 or 3 goods, but got {self.goods}.'
            raise ValueError(error_msg)
        rng = np.random.default_rng(self.random_state)
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        locations = [cells[index] for index in rng.integers(len(cells), size=self.n_traders)]
        weights = rng.dirichlet(np.ones(self.goods), size=self.n_traders)
        holdings = rng.uniform(self.low, self.high, (self.n_traders, self.goods))
        values: dict[str, Array] = {GOODS[index]: holdings[:, index] for index in range(self.goods)}
        values |= {WEIGHTS[index]: weights[:, index] for index in range(self.goods)}
        traders = GroupElement(
            'traders',
            element=GoodsTraderElement,
            size=self.n_traders,
            values=values,
            variable=True,
            location=locations,
        )
        return CompositeSystem(
            self.label,
            elements=[traders],
            interactions=[
                MovementInteraction('wander', target='traders', probability=self.mobility, phase='move'),
                ExchangeInteraction(
                    'exchange',
                    source='traders',
                    target='traders',
                    goods=GOODS[: self.goods],
                    pairing='neighbor-pairs',
                    quantity=self.quantity,
                ),
            ],
            space=LatticeSpace((self.width, self.height), boundary='periodic', metric='chebyshev'),
            time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('move'), PhaseSpec('trade')])),
        )
