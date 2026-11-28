"""Traffic systems: cars on roads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import ZZ, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role


class RoadElement(Element):
    """Cell of the Nagel-Schreckenberg road.

    It holds the velocity of the car on it, or -1 when it is empty. The local rules of the road make cars accelerate by
    one up to the largest velocity, brake to the gap ahead, slow down by one with probability `slowdown`, and move
    forward by their velocity, every step.

    Args:
        label:
            The label of the cell or the lattice group.

        velocity:
            The initial velocity, or -1 for an empty cell.

        slowdown:
            The probability of random braking.

        space:
            The lattice of the cells of the road, or `None` for one cell.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import RoadElement
        >>> RoadElement('cell').slowdown
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        velocity: Annotated[object, ZZ] = -1,
        slowdown: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.velocity = velocity
        self.slowdown = slowdown


class NagelSchreckenbergInteraction(Interaction):
    """Driving rules of the Nagel-Schreckenberg road: accelerate, brake, slow down, and move.

    Every step, a car accelerates by one up to `vmax`, brakes to the gap ahead, slows down by one with the probability
    `slowdown` of its cell, and then moves forward by its velocity. Both rules read the ordered cells ahead or behind
    up to `vmax` cells away. The movement records every jam as a `jam` event, the stopped cars connected in space and
    time after the cars move, which a jam follows as it moves back one cell per step.

    Args:
        label:
            The label of the interaction.

        target:
            The cells of the road whose state `velocity` the rules update.

        vmax:
            The largest velocity, in cells per step.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import NagelSchreckenbergInteraction
        >>> NagelSchreckenbergInteraction('driving', target='road', vmax=3).vmax
        3
    """

    def __init__(self, label: str, *, target: Role = 'road', vmax: int = 5) -> None:
        super().__init__(label)
        self.target = target
        self.vmax = vmax

    @staticmethod
    def _accelerate(velocity: Array, *ahead: Array, slowdown: Array | float, random: Array, **kwargs: object) -> Array:
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
    def _move(*behind: Array, **kwargs: object) -> Array:
        """Return the velocities after every car moves forward by its velocity."""
        following = np.full(behind[0].shape, -1.0)
        for distance, velocity in enumerate(behind):
            following = np.where(velocity == distance, velocity, following)
        return following

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the acceleration and the movement of the cars as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        velocity = context.target.velocity
        ahead = LatticeNeighborhood(tuple((distance,) for distance in range(self.vmax + 1)))
        behind = LatticeNeighborhood(tuple((-distance,) for distance in range(self.vmax + 1)))
        self.laws_: list[BaseLaw] = [
            UpdateLaw(velocity, self._accelerate, neighborhood=ahead, label='acceleration', stochastic=True),
            UpdateLaw(velocity, self._move, neighborhood=behind, activity=where(velocity == 0, 1, 0), label='jam'),
        ]


class NagelSchreckenbergSystem(BaseSystem):
    """The Nagel-Schreckenberg traffic system on a ring road.

    Jams emerge above a critical density. Every jam is recorded as a `jam` event, the stopped cars connected in space
    and time, with its size, the number of stopped car steps, its duration, its area, the number of cells it covers, and
    its profile, the stopped cars of every step. A jam still standing when the run ends is left out. The system has the
    velocities `road.velocity`, -1 on empty cells.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.engineering.traffic import NagelSchreckenbergSystem
        >>> system = NagelSchreckenbergSystem(100, density=0.1).prepare().system_
        >>> int((system.elements[0].velocity >= 0).sum())
        10
        >>> trajectory = NagelSchreckenbergSystem(200, density=0.3).prepare().simulate(until=50, random_state=0)
        >>> {event.label for event in trajectory.events}
        {'jam'}
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.length = length
        self.density = density
        self.slowdown = slowdown
        self.vmax = vmax
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        velocity = np.full(self.length, -1.0)
        velocity[rng.choice(self.length, size=round(self.density * self.length), replace=False)] = 0.0
        road = RoadElement('road', velocity=velocity, slowdown=self.slowdown, space=LatticeSpace(self.length))
        self.system_ = System(
            self.label,
            elements=[road],
            interactions=[NagelSchreckenbergInteraction('driving', target='road', vmax=self.vmax)],
        )
