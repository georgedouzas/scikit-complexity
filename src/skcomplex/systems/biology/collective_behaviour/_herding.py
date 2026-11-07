"""Collective behaviour systems: a shepherd that herds a flock of sheep to a goal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace, DiscreteTime
from ....core.components.system import System
from ..._catalog import register_system
from ._herded_sheep import HerdedSheepElement
from ._shepherd import ShepherdElement
from ._shepherding import ShepherdingInteraction


@register_system(
    field='biology/collective_behaviour',
    tags=('shepherding', 'herding', 'agents', 'swarm control'),
    references=('Strömbom et al. 2014', 'Long et al. 2020'),
    defaults={'until': 300},
)
def shepherding(
    n_sheep: int = 30,
    *,
    size: float = 100.0,
    goal: tuple[float, float] = (5.0, 5.0),
    seed: int | None = 0,
) -> System:
    """Return a shepherd that collects a scattered flock and drives it to a goal, the heuristic of Strömbom et al.

    The sheep start scattered in the middle of a square field and the shepherd behind them, opposite the goal. Sheep
    flee a near shepherd while they keep together and apart, and graze otherwise. The shepherd alternates between
    collecting the sheep furthest from the flock and driving the flock from behind towards the goal, and the zig-zag
    of this alternation brings the flock to the goal in a few hundred steps.

    Args:
        n_sheep:
            The number of sheep.

        size:
            The side of the square field.

        goal:
            The position of the goal.

        seed:
            The seed of the initial positions.

    Returns:
        system:
            The system, with the agent groups `sheep` and `shepherds`, the latter of one shepherd.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import shepherding
        >>> [group.label for group in shepherding(10).elements]
        ['sheep', 'shepherds']
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.4 * size, 0.7 * size, (n_sheep, 2))
    sheep = GroupElement(
        'sheep',
        element=HerdedSheepElement,
        size=n_sheep,
        location=[tuple(map(float, position)) for position in positions],
    )
    dogs = GroupElement(
        'shepherds',
        element=ShepherdElement,
        size=1,
        location=[(0.9 * size, 0.9 * size)],
    )
    herding = ShepherdingInteraction(
        'herding',
        sheep='sheep',
        shepherds='shepherds',
        goal=goal,
        n=max(1, round(0.9 * n_sheep)),
    )
    return System(
        elements=[sheep, dogs],
        interactions=[herding],
        space=ContinuousSpace(extent=(size, size)),
        time=DiscreteTime(),
    )
