"""Finance systems: a stock market whose fundamentalists and chartists trade in a limit order book and imitate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import ImitationInteraction, MapInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import System
from ....core.specs import (
    ContinuousMatchingSpec,
    ObservableSpec,
    PhaseSpec,
    PhasesScheduleSpec,
)
from ..._catalog import register_system
from ._book import BookInteraction
from ._stock_trader import StockTraderElement

HOLDINGS = ('money', 'shares')


def _order_book_observables(system: System) -> list[ObservableSpec]:
    """Return the share of chartists, the mean wealth of the traders, and their total money and shares."""
    del system
    return [
        ObservableSpec('chartists', group='traders', variable='strategy', reduce='mean'),
        ObservableSpec('mean_wealth', group='traders', variable='wealth', reduce='mean'),
        *(ObservableSpec(f'total_{name}', group='traders', variable=name, reduce='sum') for name in HOLDINGS),
    ]


@register_system(
    field='economics/finance',
    tags=('stock market', 'agents', 'order book', 'imitation', 'fundamentalists', 'chartists'),
    references=('Caiani et al. 2016 ch. 3', 'Lux and Marchesi 1999'),
    defaults={'until': 200, 'invariants': {f'total_{name}': 1e-6 for name in HOLDINGS}},
    observables=_order_book_observables,
)
def order_book(
    width: int = 20,
    height: int = 20,
    *,
    chartists: float = 0.3,
    fundamental: float = 100.0,
    reversion: float = 0.2,
    extrapolation: float = 1.0,
    noise: float = 0.02,
    imitation: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a stock market of fundamentalists and chartists on a grid who trade in a limit order book and imitate.

    Every step every trader forecasts the price, as a fundamentalist who expects it to revert to the fundamental value
    or as a chartist who expects the last change to go on, and quotes a limit order for one share at its forecast. The
    order book matches the orders with price-time priority, and every trade moves money and a share between the two
    traders. Then traders copy the strategy of a neighbour whose wealth grew more. Fundamentalists hold the price near
    the fundamental value, chartists amplify its swings, and imitation lets the share of chartists wander, so the
    returns come in bursts with fat tails (Caiani et al. 2016 ch. 3).

    Args:
        width:
            The number of columns of the grid, with one trader per cell.

        height:
            The number of rows of the grid.

        chartists:
            The initial share of chartists.

        fundamental:
            The fundamental value of the share.

        reversion:
            The share of the gap to the fundamental value a fundamentalist expects to close.

        extrapolation:
            The share of the last change of the price a chartist expects to repeat.

        noise:
            The standard deviation of the relative noise of a forecast.

        imitation:
            The probability that a trader looks at a neighbour in a step.

        random_state:
            The seed or generator of the initial strategies.

    Returns:
        system:
            The system, with the group `traders` and the book `book`, whose reporters are `book.price` and
            `book.volume`, and the imitation `imitation` of the strategies of more successful neighbours.

    Examples:
        >>> from skcomplex.systems.economics.finance import order_book
        >>> order_book(5, 5).elements[0].prepare().size_
        25
    """
    rng = np.random.default_rng(random_state)
    cells = [(x, y) for x in range(width) for y in range(height)]
    traders = GroupElement(
        'traders',
        element=StockTraderElement,
        size=len(cells),
        values={
            'strategy': (rng.random(len(cells)) < chartists).astype(float),
            'money': 10 * fundamental,
            'shares': 10.0,
            'wealth': 20 * fundamental,
            'fundamental': fundamental,
            'reversion': reversion,
            'extrapolation': extrapolation,
            'noise': noise,
        },
        location=cells,
    )
    accounts = [
        MapInteraction(
            f'{name}_account',
            target='traders',
            state=name,
            function=function,
            form='agent',
            phase='account',
        )
        for name, function in (('gain', StockTraderElement.get_gain), ('wealth', StockTraderElement.get_wealth))
    ]
    return System(
        elements=[traders],
        interactions=[
            BookInteraction(
                'book',
                requesters='traders',
                providers='traders',
                rule=ContinuousMatchingSpec(),
                opening=fundamental,
            ),
            *accounts,
            ImitationInteraction(
                'imitation',
                target='traders',
                state='strategy',
                payoff='gain',
                rule='better',
                form='agent',
                probability=imitation,
                phase='imitation',
            ),
        ],
        space=LatticeSpace((width, height), boundary='periodic', neighborhood='von_neumann'),
        time=DiscreteTime(
            schedule=PhasesScheduleSpec(
                [
                    PhaseSpec('quote'),
                    PhaseSpec('trade'),
                    PhaseSpec('account'),
                    PhaseSpec('imitation', activation='simultaneous'),
                ],
            ),
        ),
    )
