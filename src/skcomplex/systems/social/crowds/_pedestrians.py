"""Pedestrian systems: walkers driven by desired velocities and social forces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement, PointElement
from ....core.components.interactions import (
    BaseInteraction,
)
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ..._catalog import register_system
from ._motivation import MotivationInteraction
from ._social_repulsion import SocialRepulsionInteraction
from ._walls import WallsInteraction


@register_system(
    field='social/crowds',
    tags=('pedestrians', 'particles', 'lanes', 'self-organization'),
    references=('Helbing and Molnár 1995', 'Helbing 2001'),
    defaults={'until': 20.0, 'dt': 0.2, 'solver': RK4Solver(dt=0.05)},
)
def social_force(
    n_walkers: int = 20,
    *,
    length: float = 10.0,
    speed: float = 1.34,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return two crowds walking in opposite directions along a periodic corridor, which form lanes.

    Args:
        n_walkers:
            The number of walkers in each direction.

        length:
            The length and the width of the corridor, periodic along its length.

        speed:
            The desired speed.

        random_state:
            The seed or generator of the initial positions.

    Returns:
        system:
            The system, with the groups `rightward` and `leftward`.

    Examples:
        >>> from skcomplex.systems.social.crowds import social_force
        >>> [group.size for group in social_force(5).elements]
        [5, 5]
    """
    rng = np.random.default_rng(random_state)
    cutoff = 2.0
    walkers = [
        GroupElement(
            label,
            element=PointElement,
            size=n_walkers,
            values={'x': rng.uniform([0.0, 0.5], [length, length - 0.5], (n_walkers, 2))},
        )
        for label in ('rightward', 'leftward')
    ]
    interactions: list[BaseInteraction] = []
    for label, sign in (('rightward', 1.0), ('leftward', -1.0)):
        interactions += [
            MotivationInteraction(f'{label}_drive', target=label, speed=speed, direction=[sign, 0.0]),
            WallsInteraction(f'{label}_walls', target=label, width=length),
        ]
    pairs = (('rightward', 'rightward'), ('leftward', 'leftward'), ('rightward', 'leftward'))
    interactions += [
        SocialRepulsionInteraction(f'{first}_{second}', first=first, second=second, cutoff=cutoff, period=length)
        for first, second in pairs
    ]
    return System(
        elements=walkers,
        interactions=interactions,
        space=ContinuousSpace(extent=(length, length), boundary='periodic'),
    )
