"""Markets systems: shops on a grid that compete for nearby shoppers, and the digital market that removes distance."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import (
    LifeCycleInteraction,
    MapInteraction,
    MatchingInteraction,
    SignAdjustmentInteraction,
)
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import SystemView
from ....core.specs import (
    ObservableSpec,
    PartnerSpec,
    PhaseSpec,
    PhasesScheduleSpec,
)
from ..._catalog import register_system
from ._resident import ResidentElement
from ._store import StoreElement


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


@register_system(
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
            The system, with the groups `shoppers` and `shops` and the market `shopping`, whose reporters are the
            `shopping.value`, `shopping.volume`, and `shopping.unmatched` of the purchases.

    Examples:
        >>> from skcomplex.systems.economics.markets import shops
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
        LifeCycleInteraction('closure', target='shops', death=lambda agent, system: agent.money < 0, phase='accounts'),
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
        elements=[shoppers, stores],
        interactions=[*plan, shopping, *settlement],
        space=LatticeSpace((width, height), boundary='fixed', neighborhood='moore'),
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('market'), PhaseSpec('accounts')])),
    )
