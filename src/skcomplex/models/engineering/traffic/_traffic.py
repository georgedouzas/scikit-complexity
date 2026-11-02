"""Traffic models: cars on roads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, LatticeElement
from ....core.interactions import LocalRuleInteraction
from ....core.spaces import LatticeNeighborhood, LatticeSpace
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model


def _accelerate(velocity: Array, *ahead: Array, slowdown: Array | float, random: Array) -> Array:
    """Return the velocities after the cars accelerate, brake to the gap ahead, and slow down at random."""
    vmax = len(ahead)
    gap = np.full(velocity.shape, float(vmax))
    for distance in range(vmax, 0, -1):
        gap = np.where(ahead[distance - 1] >= 0, distance - 1, gap)
    following = np.minimum(np.minimum(velocity + 1, vmax), gap)
    following = np.where(random < slowdown, np.maximum(following - 1, 0), following)
    return np.where(velocity >= 0, following, -1.0)


def _move(*behind: Array) -> Array:
    """Return the velocities after every car moves forward by its velocity."""
    following = np.full(behind[0].shape, -1.0)
    for distance, velocity in enumerate(behind):
        following = np.where(velocity == distance, velocity, following)
    return following


class RoadElement(Element):
    """Cell of the Nagel-Schreckenberg road: the velocity of the car on it, or -1 when it is empty.

    The local rules of the road make cars accelerate by one up to the largest velocity, brake to the gap ahead, slow
    down by one with probability `slowdown`, and move forward by their velocity, every step.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        velocity:
            The initial velocity, or -1 for an empty cell.

        slowdown:
            The probability of random braking.

    Examples:
        >>> from skcomplex.models.engineering.traffic import RoadElement
        >>> RoadElement('cell').slowdown
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        velocity: Annotated[float, StateSpec(domain=ZZ)] = -1,
        slowdown: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.velocity = velocity
        self.slowdown = slowdown


@register_model(
    field='engineering/traffic',
    tags=('lattice', 'traffic', 'jams', 'stochastic'),
    references=('Nagel and Schreckenberg 1992', 'Charbonneau 2017'),
    defaults={'until': 200, 'dt': 1},
)
def nagel_schreckenberg(
    length: int = 1000,
    *,
    density: float = 0.2,
    slowdown: float = 0.3,
    vmax: int = 5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Nagel-Schreckenberg traffic model on a ring road, where jams emerge above a critical density.

    Args:
        length:
            The number of cells of the road.

        density:
            The fraction of cells with a car.

        slowdown:
            The probability of random braking.

        vmax:
            The largest velocity.

        random_state:
            The seed or generator of the initial positions of the cars.

    Returns:
        system:
            The model, with the velocities `road.velocity`, -1 on empty cells.

    Examples:
        >>> from skcomplex.models.engineering.traffic import nagel_schreckenberg
        >>> int((nagel_schreckenberg(100, density=0.1).elements[0].values['velocity'] >= 0).sum())
        10
    """
    rng = np.random.default_rng(random_state)
    velocity = np.full(length, -1.0)
    velocity[rng.choice(length, size=round(density * length), replace=False)] = 0.0
    road = LatticeElement(
        'road',
        element=RoadElement,
        space=LatticeSpace(length),
        values={'velocity': velocity, 'slowdown': slowdown},
    )
    ahead = LatticeNeighborhood(tuple((distance,) for distance in range(vmax + 1)))
    behind = LatticeNeighborhood(tuple((-distance,) for distance in range(vmax + 1)))
    return System(
        elements=[road],
        interactions=[
            LocalRuleInteraction(
                'acceleration',
                target='road',
                state='velocity',
                rule=_accelerate,
                neighborhood=ahead,
                stochastic=True,
            ),
            LocalRuleInteraction('movement', target='road', state='velocity', rule=_move, neighborhood=behind),
        ],
    )
