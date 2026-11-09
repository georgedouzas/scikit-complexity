"""Traffic systems: cars on roads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import LatticeNeighborhood, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array


class RoadElement(Element):
    """Cell of the Nagel-Schreckenberg road.

    It holds the velocity of the car on it, or -1 when it is empty. The local rules of the road make cars accelerate by
    one up to the largest velocity, brake to the gap ahead, slow down by one with probability `slowdown`, and move
    forward by their velocity, every step.

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
        >>> from skcomplex.systems.engineering.traffic import RoadElement
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


class NagelSchreckenbergSystem(BaseSystem):
    """The Nagel-Schreckenberg traffic system on a ring road.

    Jams emerge above a critical density. The system has the velocities `road.velocity`, -1 on empty cells.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import NagelSchreckenbergSystem
        >>> system = NagelSchreckenbergSystem(100, density=0.1).prepare().system_
        >>> int((system.elements[0].values['velocity'] >= 0).sum())
        10
    """

    field = 'engineering/traffic'
    topics = ('lattice', 'traffic', 'jams', 'stochastic')
    references = ('Nagel and Schreckenberg 1992', 'Charbonneau 2017')
    until = 200
    dt = 1

    def __init__(
        self,
        length: int = 1000,
        *,
        density: float = 0.2,
        slowdown: float = 0.3,
        vmax: int = 5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.length = length
        self.density = density
        self.slowdown = slowdown
        self.vmax = vmax
        self.random_state = random_state

    @staticmethod
    def _accelerate(velocity: Array, *ahead: Array, slowdown: Array | float, random: Array) -> Array:
        """Return the velocities after the cars accelerate, brake, and slow down at random.

        The cars brake to the gap ahead.
        """
        vmax = len(ahead)
        gap = np.full(velocity.shape, float(vmax))
        for distance in range(vmax, 0, -1):
            gap = np.where(ahead[distance - 1] >= 0, distance - 1, gap)
        following = np.minimum(np.minimum(velocity + 1, vmax), gap)
        following = np.where(random < slowdown, np.maximum(following - 1, 0), following)
        return np.where(velocity >= 0, following, -1.0)

    @staticmethod
    def _move(*behind: Array) -> Array:
        """Return the velocities after every car moves forward by its velocity."""
        following = np.full(behind[0].shape, -1.0)
        for distance, velocity in enumerate(behind):
            following = np.where(velocity == distance, velocity, following)
        return following

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        velocity = np.full(self.length, -1.0)
        velocity[rng.choice(self.length, size=round(self.density * self.length), replace=False)] = 0.0
        road = GroupElement(
            'road',
            element=RoadElement,
            space=LatticeSpace(self.length),
            values={'velocity': velocity, 'slowdown': self.slowdown},
            on='sites',
        )
        ahead = LatticeNeighborhood(tuple((distance,) for distance in range(self.vmax + 1)))
        behind = LatticeNeighborhood(tuple((-distance,) for distance in range(self.vmax + 1)))
        return CompositeSystem(
            self.label,
            elements=[road],
            interactions=[
                LocalRuleInteraction(
                    'acceleration',
                    target='road',
                    state='velocity',
                    rule=self._accelerate,
                    neighborhood=ahead,
                    stochastic=True,
                ),
                LocalRuleInteraction('movement', target='road', state='velocity', rule=self._move, neighborhood=behind),
            ],
        )
