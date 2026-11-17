"""Spring systems.

The module holds masses on springs and dampers between walls.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ..common.particles import LinearDragInteraction, RestoringForceInteraction


class ElasticLinkInteraction(Interaction):
    """Spring of stiffness `k` and rest length `length` between two masses on a line.

    The energy is `k (|x1 - x2| - length)² / 2`, so the masses follow `m1 x1'' = f` and `m2 x2'' = -f`, with the force
    `f = -k (|x1 - x2| - length) (x1 - x2) / |x1 - x2|`. It is the spring between two carts, the bond of a diatomic
    molecule, or the coupling of two floors of a building.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first mass.

        second:
            The path of the second mass.

        k:
            The stiffness.

        length:
            The rest length.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.mechanics import ElasticLinkInteraction
        >>> left, right = Element('left', values={'x': 0.0, 'm': 1.0}), Element('right', values={'x': 2.0, 'm': 1.0})
        >>> link = ElasticLinkInteraction('bond', first='left', second='right', k=1.0, length=1.0)
        >>> trajectory = System(elements=[left, right], interactions=[link]).prepare().simulate(until=1.0, dt=1.0)
        >>> round(float(trajectory.get_values('left.x')[-1] + trajectory.get_values('right.x')[-1]), 6)
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        first: Role,
        second: Role,
        k: Annotated[float, RealSet.open(0, oo)] = 1.0,
        length: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.k = k
        self.length = length

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the accelerations of both masses.

        Args:
            context:
                The namespace of the two masses, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            t, first, second = context.t, context.first, context.second
            gap = first.x - second.x
            distance = sqrt(gap**2)
            force = -context.k * (distance - context.length) * gap / distance
            self.laws_ = [RateLaw(first.x.diff(t, 2), force / first.m), RateLaw(second.x.diff(t, 2), -force / second.m)]


class PeriodicForceInteraction(Interaction):
    """Periodic force `F cos(omega t)` on a mass `m`.

    Every coordinate `x` of the target follows `x'' = F cos(omega t) / m`. It is the motor that shakes a driven
    oscillator, an alternating voltage on a circuit, or the push of a parent on a swing.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the mass.

        F:
            The amplitude of the force.

        omega:
            The angular frequency of the force.

        coordinates:
            The names of the coordinates of the target that the force drives.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.mechanics import PeriodicForceInteraction
        >>> ball = Element('ball', values={'x': 0.0, 'm': 2.0})
        >>> force = PeriodicForceInteraction('drive', target='ball', F=2.0, omega=1.0)
        >>> trajectory = System(elements=[ball], interactions=[force]).prepare().simulate(until=1.0, dt=1.0)
        >>> round(float(trajectory.get_values('ball.x')[-1]), 3)
        0.46
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        F: float = 0.0,
        omega: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        coordinates: Sequence[str] = ('x',),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.F = F
        self.omega = omega
        self.coordinates = coordinates

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of every coordinate of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            t, target = context.t, context.target
            force = context.F * cos(context.omega * t)
            self.laws_ = [RateLaw(getattr(target, name).diff(t, 2), force / target.m) for name in self.coordinates]


class TwoMassSpringDamperSystem(BaseSystem):
    """Two masses between two walls, with dampers and forces.

    Springs tie the masses to the walls and to each other.

    The first mass rests at `x01 = l1` and the second at `x02 = l1 + l2`. They follow `m1 x1'' = -k1 (x1 - x01) + k ((x2
    - x1) - (x02 - x01)) - b1 x1' + F1 cos(ω1 t)` and `m2 x2'' = -k ((x2 - x1) - (x02 - x01)) - k2 (x2 - x02) - b2 x2' +
    F2 cos(ω2 t)`. The system has the masses `first` and `second` and their positions `first.x` and `second.x`.

    Args:
        x1:
            The initial position of the first mass.

        x2:
            The initial position of the second mass.

        m1:
            The first mass.

        m2:
            The second mass.

        k1:
            The stiffness of the spring between the left wall and the first mass.

        k2:
            The stiffness of the spring between the second mass and the right wall.

        k:
            The stiffness of the spring between the masses.

        b1:
            The damping coefficient of the first mass.

        b2:
            The damping coefficient of the second mass.

        l1:
            The rest length of the first spring.

        l2:
            The rest length of the middle spring.

        F1:
            The amplitude of the periodic force on the first mass.

        F2:
            The amplitude of the periodic force on the second mass.

        omega1:
            The angular frequency of the force on the first mass.

        omega2:
            The angular frequency of the force on the second mass.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import TwoMassSpringDamperSystem
        >>> TwoMassSpringDamperSystem(l1=2.0).prepare().system_.interactions[0].center
        2.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'oscillation', 'normal modes', 'forcing')
    references = ('Ryzhov et al. 2021',)
    until = 30.0
    dt = 0.05

    def __init__(
        self,
        x1: float = 1.2,
        x2: float = 2.0,
        *,
        m1: float = 1.0,
        m2: float = 1.0,
        k1: float = 1.0,
        k2: float = 1.0,
        k: float = 1.0,
        b1: float = 0.0,
        b2: float = 0.0,
        l1: float = 1.0,
        l2: float = 1.0,
        F1: float = 0.0,
        F2: float = 0.0,
        omega1: float = 1.0,
        omega2: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x1 = x1
        self.x2 = x2
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.k = k
        self.b1 = b1
        self.b2 = b2
        self.l1 = l1
        self.l2 = l2
        self.F1 = F1
        self.F2 = F2
        self.omega1 = omega1
        self.omega2 = omega2

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of two masses between two walls, with springs, dampers, and forces.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rest1, rest2 = self.l1, self.l1 + self.l2
        self.system_ = System(
            self.label,
            elements=[
                Element('first', values={'x': self.x1, 'm': self.m1}),
                Element('second', values={'x': self.x2, 'm': self.m2}),
            ],
            interactions=[
                RestoringForceInteraction('left_spring', target='first', k=self.k1, center=rest1),
                RestoringForceInteraction('right_spring', target='second', k=self.k2, center=rest2),
                ElasticLinkInteraction('middle_spring', first='first', second='second', k=self.k, length=self.l2),
                LinearDragInteraction('first_damper', target='first', gamma=self.b1),
                LinearDragInteraction('second_damper', target='second', gamma=self.b2),
                PeriodicForceInteraction('first_force', target='first', F=self.F1, omega=self.omega1),
                PeriodicForceInteraction('second_force', target='second', F=self.F2, omega=self.omega2),
            ],
        )
