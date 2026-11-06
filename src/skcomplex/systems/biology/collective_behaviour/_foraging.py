"""Collective behaviour systems: animals that coordinate through signals."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import FieldElement, GroupElement
from ....core.components.interactions import (
    DiffusionInteraction,
    MapInteraction,
    MovementInteraction,
    TransferInteraction,
)
from ....core.components.laws import Source
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ..._catalog import register_system
from ..common.stigmergy import EvaporationInteraction
from ._ant import AntElement
from ._ant_view import AntView
from ._pheromone import PheromoneElement

PHEROMONE = 'ground/pheromone.level'


def _get_diffusion(namespace: SimpleNamespace) -> Expression:
    """Return the diffusion coefficient `D` of the pheromone."""
    coefficient: Expression = namespace.target.D
    return coefficient


def _get_nest(ant: AntView) -> tuple[int, int]:
    """Return the cell of the nest of an ant."""
    return round(ant.nest_x), round(ant.nest_y)


def _homes(ant: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether an ant carries food away from the nest, so it walks home."""
    del system
    return isinstance(ant, AntView) and bool(ant.carrying) and ant.location != _get_nest(ant)


def _searches(ant: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether an ant without food stands farther than `food_radius` from the food, so it searches."""
    del system
    if not isinstance(ant, AntView) or ant.carrying:
        return False
    x, y = cast('tuple[int, int]', ant.location)
    return bool(np.hypot(x - ant.food_x, y - ant.food_y) > ant.food_radius)


def _get_home_step(ant: AgentViewProtocol, system: SystemViewProtocol) -> object:
    """Return the cell one step straight toward the nest."""
    del system
    if not isinstance(ant, AntView):
        return None
    x, y = cast('tuple[int, int]', ant.location)
    nest = _get_nest(ant)
    return x + int(np.sign(nest[0] - x)), y + int(np.sign(nest[1] - y))


def _get_search_step(ant: AgentViewProtocol, system: SystemViewProtocol) -> object:
    """Return a neighbouring cell drawn with a weight that grows with its pheromone and its distance from the nest."""
    if not isinstance(ant, AntView):
        return None
    x, y = cast('tuple[int, int]', ant.location)
    nest = _get_nest(ant)
    cells = cast('list[tuple[int, int]]', list(ant.neighbor_cells()))
    distance = np.hypot(x - nest[0], y - nest[1])
    weights = np.asarray(
        [
            (1.0 + ant.attraction * system.field(PHEROMONE, cell))
            * np.exp(ant.outward * (np.hypot(cell[0] - nest[0], cell[1] - nest[1]) - distance))
            for cell in cells
        ],
    )
    return cells[int(system.rng.choice(len(cells), p=weights / weights.sum()))]


def _carries(ant: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> bool:
    """Return whether an ant carries food and walked home in its activation, so it marks its cell with pheromone."""
    del cell, system
    return isinstance(ant, AntView) and bool(ant.carrying) and not ant.arrived


def _get_deposit(ant: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
    """Return the pheromone `deposit` an ant leaves on its cell."""
    del cell, system
    return float(ant.deposit) if isinstance(ant, AntView) else 0.0


def _get_carrying(ant: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return whether an ant carries food next: the step after it arrives, it picks food up or drops it at the nest."""
    if not isinstance(ant, AntView):
        return 0.0
    return float(not ant.carrying) if ant.arrived else float(ant.carrying)


def _get_arrived(ant: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return whether an ant stands at the nest with food, or near the food without it, at the end of its activation."""
    if not isinstance(ant, AntView):
        return 0.0
    return float(not _homes(ant, system) and not _searches(ant, system))


@register_system(
    field='biology/collective_behaviour',
    tags=('field', 'agents', 'stigmergy', 'co-simulation'),
    references=('Argun et al. 2021', 'Carmichael and Hadžikadić 2019'),
    defaults={'until': 600, 'dt': 1},
)
def pheromone_foraging(
    width: int = 40,
    height: int = 20,
    *,
    n_ants: int = 40,
    D: float = 0.05,
    evaporation: float = 0.05,
    attraction: float = 5.0,
) -> System:
    """Return ants that find food and lay a pheromone trail home, which other ants follow: stigmergy.

    The ants live on a grid and the pheromone on a lattice of the same shape, as two partitions of a co-simulation
    that exchange the field and the deposits at every step. The nest is near the left edge and the food near the right.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        n_ants:
            The number of ants, which start at the nest.

        D:
            The diffusion of the pheromone.

        evaporation:
            The evaporation rate of the pheromone.

        attraction:
            How strongly pheromone draws searching ants.

    Returns:
        system:
            The system, with the ants `colony/ants` and the pheromone `ground/pheromone.level`.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import pheromone_foraging
        >>> [child.label for child in pheromone_foraging().systems]
        ['colony', 'ground']
    """
    nest, food = (5, height // 2), (width - 6, height // 2)
    ants = GroupElement(
        'ants',
        element=AntElement,
        size=n_ants,
        values={
            'nest_x': float(nest[0]),
            'nest_y': float(nest[1]),
            'food_x': float(food[0]),
            'food_y': float(food[1]),
            'attraction': attraction,
        },
        location=nest,
    )
    pheromone = FieldElement(
        'pheromone',
        element=PheromoneElement,
        space=LatticeSpace((width, height)),
        boundary={'level': 'zero-flux'},
        values={'D': D, 'evaporation': evaporation},
    )
    return System(
        systems=[
            System('colony', elements=[ants], space=LatticeSpace((width, height), boundary='fixed')),
            System(
                'ground',
                elements=[pheromone],
                interactions=[
                    DiffusionInteraction('diffusion', target='pheromone', state='level', coefficient=_get_diffusion),
                    EvaporationInteraction('evaporation', target='pheromone', state='level'),
                ],
            ),
        ],
        interactions=[
            MovementInteraction(
                'homing',
                target='colony/ants',
                step=_get_home_step,
                condition=_homes,
                schedule='member',
            ),
            TransferInteraction(
                'marking',
                source='colony/ants',
                target='ground/pheromone',
                state=Source(),
                into='level',
                amount=_get_deposit,
                pairing='site',
                guard=_carries,
                form='agent',
                schedule='member',
            ),
            MovementInteraction(
                'search',
                target='colony/ants',
                step=_get_search_step,
                condition=_searches,
                schedule='member',
            ),
            MapInteraction(
                'carrying',
                target='colony/ants',
                state='carrying',
                function=_get_carrying,
                form='agent',
                schedule='member',
            ),
            MapInteraction(
                'arrival',
                target='colony/ants',
                state='arrived',
                function=_get_arrived,
                form='agent',
                schedule='member',
            ),
        ],
    )
