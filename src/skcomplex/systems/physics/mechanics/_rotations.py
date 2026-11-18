"""Rotation systems.

The module covers a rotating swing, balls on rotating rods, and a gyroscope in a gimbal.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, oo, sin
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, Period
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ..common.mechanics import LagrangianInteraction


class GyroscopeElement(Element):
    """Gyroscope in a gimbal.

    A rotor in an inner ring turns in an outer ring about a fixed axis.

    The angle `alpha` of the outer ring, `beta` of the inner ring, and `gamma` of the rotor are the coordinates of
    `GyroscopeInteraction`, and their angular velocities are `alpha'`, `beta'`, and `gamma'`.

    Args:
        label:
            The label of the gyroscope.

        alpha:
            The initial angle of the outer ring.


        beta:
            The initial angle of the inner ring.


        gamma:
            The initial angle of the rotor.


        A:
            The equatorial moment of inertia of the rotor.

        C:
            The polar moment of inertia of the rotor.

        A1:
            The first principal moment of inertia of the inner ring.

        B1:
            The second principal moment of inertia of the inner ring.

        C1:
            The third principal moment of inertia of the inner ring.

        A2:
            The moment of inertia of the outer ring about its axis.

        values:
            Further values by name, such as an initial velocity named with a prime, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import GyroscopeElement
        >>> GyroscopeElement('gyroscope', C=3.0).C
        3.0
    """

    def __init__(
        self,
        label: str,
        *,
        alpha: Annotated[float, Period(2 * math.pi)] = 0.0,
        beta: float = 0.3,
        gamma: Annotated[float, Period(2 * math.pi)] = 0.0,
        A: Annotated[float, RealSet.open(0, oo)] = 1.0,
        C: Annotated[float, RealSet.open(0, oo)] = 2.0,
        A1: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        B1: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        C1: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        A2: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.A = A
        self.C = C
        self.A1 = A1
        self.B1 = B1
        self.C1 = C1
        self.A2 = A2


class RotatingBallElement(Element):
    """Ball of mass `m` on a rod at the distance `r` from the joint.

    The rod turns about a vertical axis by the angle `theta`.

    The rod is fixed at the angle `beta` to the axis, so the ball is at the distance `r sin β` from it, and `H` is its
    angular momentum about the axis.

    Args:
        label:
            The label of the ball.

        theta:
            The initial angle about the axis.

        m:
            The mass of the ball.

        r:
            The distance of the ball from the joint of the rod.

        beta:
            The angle of the rod to the axis, in radians.

        H:
            The angular momentum of the ball about the axis.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingBallElement
        >>> RotatingBallElement('ball', H=2.0).H
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        theta: Annotated[float, Period(2 * math.pi)] = 0.0,
        m: Annotated[float, RealSet.open(0, oo)] = 1.0,
        r: Annotated[float, RealSet.open(0, oo)] = 1.0,
        beta: Annotated[float, RealSet.open(0, oo)] = math.pi / 2,
        H: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.theta = theta
        self.m = m
        self.r = r
        self.beta = beta
        self.H = H


class RotatingSwingElement(Element):
    """Rotating swing.

    Chairs on chains hang from the rods of a stand that turns about a vertical axis.

    The angle `phi` of the stand and the angle `theta` of the chains from the vertical are the coordinates of
    `RotatingSwingInteraction`, and their angular velocities are `phi'` and `theta'`.

    Args:
        label:
            The label of the swing.

        phi:
            The initial angle of the stand.


        theta:
            The initial angle of the chains.


        J:
            The moment of inertia of the stand.

        m:
            The mass of every chair.

        n_chairs:
            The number of chairs.

        a:
            The length of the rods.

        l:
            The length of the chains.

        g:
            The gravitational acceleration.

        values:
            Further values by name, such as an initial velocity named with a prime, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingSwingElement
        >>> RotatingSwingElement('swing', n_chairs=4.0).n_chairs
        4.0
    """

    def __init__(
        self,
        label: str,
        *,
        phi: Annotated[float, Period(2 * math.pi)] = 0.0,
        theta: float = 0.1,
        J: Annotated[float, RealSet.open(0, oo)] = 10.0,
        m: Annotated[float, RealSet.open(0, oo)] = 1.0,
        n_chairs: Annotated[float, RealSet.open(0, oo)] = 8.0,
        a: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        l: Annotated[float, RealSet.open(0, oo)] = 1.0,
        g: Annotated[float, RealSet.open(0, oo)] = 9.81,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.phi = phi
        self.theta = theta
        self.J = J
        self.m = m
        self.n_chairs = n_chairs
        self.a = a
        self.l = l
        self.g = g


class RotationInteraction(Interaction):
    """Rotation of a ball about an axis at the rate fixed by its angular momentum.

    The angle `theta` of the target follows `theta' = H / (m r² sin² β)`, with the angular momentum `H`, the mass `m`,
    the distance `r` from the joint, and the angle `β` of the rod, all values of the target.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the ball.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotationInteraction
        >>> RotationInteraction('turn', target='ball').target
        'ball'
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the angle of the ball.

        Args:
            context:
                The namespace of the ball and the time `t`.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            ball = context.target
            self.laws_ = [RateLaw(ball.theta.diff(context.t), ball.H / (ball.m * ball.r**2 * sin(ball.beta) ** 2))]


class GyroscopeInteraction(LagrangianInteraction):
    """Motion of a gimballed gyroscope.

    The motion is the Euler-Lagrange equations of its kinetic energy with a torque on `beta`.

    The Lagrangian is the kinetic energy `[A2 + (A + A1) cos² β + (C + C1) sin² β] alpha'² / 2 + (A + B1) β'² / 2
    + C alpha' gamma' sin β + C gamma'² / 2`, and the torque acts on the axis of the inner ring.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the gyroscope.

        torque:
            The torque on the axis of the inner ring.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import GyroscopeInteraction
        >>> GyroscopeInteraction('motion', target='gyroscope', torque=2.0).torque
        2.0
    """

    def __init__(self, label: str, *, target: Role, torque: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.torque = torque

    def _get_coordinates(self, context: SimpleNamespace) -> list[Expression]:
        """Return the angles of the outer ring, the inner ring, and the rotor."""
        return [context.target.alpha, context.target.beta, context.target.gamma]

    def _get_forces(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> list[Expression]:
        """Return the generalized forces.

        The torque acts on the inner ring and none on the outer ring and the rotor.
        """
        return [0, context.torque, 0]

    def _get_lagrangian(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the kinetic energy of the rings and the rotor.

        Args:
            context:
                The namespace of the gyroscope, the time `t`, and the parameters.

            positions:
                The symbols of the angles.

            velocities:
                The symbols of the angular velocities.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = context.target
        beta = positions[1]
        da, db, dc = velocities
        outer = p.A2 + (p.A + p.A1) * cos(beta) ** 2 + (p.C + p.C1) * sin(beta) ** 2
        return outer * da**2 / 2 + (p.A + p.B1) * db**2 / 2 + p.C * da * dc * sin(beta) + p.C * dc**2 / 2


class RotatingSwingInteraction(LagrangianInteraction):
    """Motion of a rotating swing.

    The motion is the Euler-Lagrange equations with a torque and friction on the stand.

    The Lagrangian is `[J + n m (a + l sin θ)²] φ'² / 2 + n m l² θ'² / 2 + n m g l cos θ`, the torque drives the angle
    `φ` of the stand, and the Rayleigh dissipation `c φ'² / 2` adds the friction of its axis.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the swing.

        torque:
            The torque that drives the stand.

        friction:
            The friction coefficient of the stand's axis.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingSwingInteraction
        >>> RotatingSwingInteraction('motion', target='swing', friction=0.5).friction
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        torque: float = 0.0,
        friction: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.torque = torque
        self.friction = friction

    def _get_coordinates(self, context: SimpleNamespace) -> list[Expression]:
        """Return the angles of the stand and the chains."""
        return [context.target.phi, context.target.theta]

    def _get_dissipation(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Rayleigh dissipation `c φ'² / 2` of the stand's axis."""
        return context.friction * velocities[0] ** 2 / 2

    def _get_forces(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> list[Expression]:
        """Return the generalized forces, the torque on the stand and none on the chains."""
        return [context.torque, 0]

    def _get_lagrangian(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Lagrangian of the stand and the chairs.

        Args:
            context:
                The namespace of the swing, the time `t`, and the parameters.

            positions:
                The symbols of the angles.

            velocities:
                The symbols of the angular velocities.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = context.target
        theta = positions[1]
        chairs = p.n_chairs * p.m
        radius = p.a + p.l * sin(theta)
        kinetic = (p.J + chairs * radius**2) * velocities[0] ** 2 / 2
        kinetic += chairs * p.l**2 * velocities[1] ** 2 / 2
        return kinetic + chairs * p.g * p.l * cos(theta)


class RotatingSwingSystem(BaseSystem):
    """A rotating swing.

    Chairs on chains hang from the rods of a stand that turns about a vertical axis.

    The stand of inertia `J` turns by the angle `φ`, and the chairs, all deflected by the same angle `θ`, follow the
    Lagrange equations `[J + n m (a + l sin θ)²] φ'' + 2 n m l φ' θ' (a + l sin θ) cos θ = M - c φ'` and `l θ'' - φ'² (a
    + l sin θ) cos θ + g sin θ = 0`. At a steady speed `ω` the chairs settle at the angle where `g tan θ = ω² (a + l sin
    θ)`. Without torque and friction the angular momentum of the swing is conserved. The system has the angle of the
    stand `swing.phi` and of the chains `swing.theta`, and their angular velocities `swing.phi'` and `swing.theta'`.

    Args:
        omega:
            The initial angular velocity of the stand.

        theta:
            The initial angle of the chains from the vertical.

        n_chairs:
            The number of chairs.

        J:
            The moment of inertia of the stand.

        m:
            The mass of every chair.

        a:
            The length of the rods.

        l:
            The length of the chains.

        g:
            The gravitational acceleration.

        torque:
            The torque that drives the stand.

        friction:
            The friction coefficient of the stand's axis.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingSwingSystem
        >>> trajectory = RotatingSwingSystem(omega=1.0).prepare().simulate(until=1.0)
        >>> float(trajectory.get_values("swing.phi'")[0])
        1.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'second order', 'rotation', 'lagrangian')
    references = ('Ryzhov et al. 2021',)
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        omega: float = 2.0,
        theta: float = 0.1,
        *,
        n_chairs: int = 8,
        J: float = 10.0,
        m: float = 1.0,
        a: float = 1.0,
        l: float = 1.0,
        g: float = 9.81,
        torque: float = 0.0,
        friction: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.omega = omega
        self.theta = theta
        self.n_chairs = n_chairs
        self.J = J
        self.m = m
        self.a = a
        self.l = l
        self.g = g
        self.torque = torque
        self.friction = friction

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a stand that turns with chairs on chains.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        swing = RotatingSwingElement(
            'swing',
            theta=self.theta,
            n_chairs=self.n_chairs,
            J=self.J,
            m=self.m,
            a=self.a,
            l=self.l,
            g=self.g,
            values={"phi'": self.omega},
        )
        motion = RotatingSwingInteraction('motion', target='swing', torque=self.torque, friction=self.friction)
        self.system_ = System(self.label, elements=[swing], interactions=[motion])


class RotatingBallsOnRodsSystem(BaseSystem):
    """Two balls on rods that turn about a vertical axis.

    The balls conserve their angular momentum.

    Each ball sits at the distance `r` along a rod fixed at the angle `β` to the axis, so its distance from the axis is
    `r sin β`. Its angular momentum `H = m ω r² sin² β` is fixed by the initial angular velocity, and the balls turn at
    `θ' = H / (m r² sin² β)`, so changing `r` during a run changes the angular velocity by `1 / r²`. The system has the
    angles `right.theta` and `left.theta` of the two balls.

    Args:
        omega:
            The initial angular velocity.

        r:
            The distance of the balls from the joint of the rods.

        beta:
            The angle of the rods to the vertical axis, in radians.

        m:
            The mass of every ball.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingBallsOnRodsSystem
        >>> RotatingBallsOnRodsSystem(omega=2.0).prepare().system_.elements[0].H
        2.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'rotation', 'angular momentum')
    references = ('Ryzhov et al. 2021',)
    until = 10.0
    dt = 0.02

    def __init__(
        self,
        omega: float = 1.0,
        *,
        r: float = 1.0,
        beta: float = math.pi / 2,
        m: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.omega = omega
        self.r = r
        self.beta = beta
        self.m = m

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of two balls on rods that turn about a vertical axis.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        momentum = self.m * self.omega * self.r**2 * math.sin(self.beta) ** 2
        self.system_ = System(
            self.label,
            elements=[
                RotatingBallElement('right', theta=0.0, m=self.m, r=self.r, beta=self.beta, H=momentum),
                RotatingBallElement('left', theta=math.pi, m=self.m, r=self.r, beta=self.beta, H=momentum),
            ],
            interactions=[RotationInteraction(f'{ball}_turn', target=ball) for ball in ('right', 'left')],
        )


class GimballedGyroscopeSystem(BaseSystem):
    """A gyroscope in a gimbal.

    A rotor in an inner ring turns in an outer ring about a fixed axis.

    The outer ring turns by `alpha` about the fixed axis, the inner ring by `beta` in the outer ring, and the rotor
    spins by `gamma` in the inner ring. The equations are the Lagrange equations of the kinetic energy `[A2 + (A + A1)
    cos² beta + (C + C1) sin² beta] alpha'² / 2 + (A + B1) beta'² / 2 + C alpha' gamma' sin beta + C gamma'² / 2`, with
    a torque on the axis of the inner ring, which makes the fast rotor precess about the fixed axis. The momenta
    conjugate to `alpha` and `gamma` are conserved. The system has the angles `gyroscope.alpha`, `gyroscope.beta`, and
    `gyroscope.gamma`, and their angular velocities, such as `gyroscope.gamma'`.

    Args:
        spin:
            The initial angular velocity of the rotor in the inner ring.

        beta:
            The initial angle of the inner ring.

        A:
            The equatorial moment of inertia of the rotor.

        C:
            The polar moment of inertia of the rotor.

        A1:
            The first principal moment of inertia of the inner ring.

        B1:
            The second principal moment of inertia of the inner ring.

        C1:
            The third principal moment of inertia of the inner ring.

        A2:
            The moment of inertia of the outer ring about its axis.

        torque:
            The torque on the axis of the inner ring.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import GimballedGyroscopeSystem
        >>> trajectory = GimballedGyroscopeSystem(spin=10.0).prepare().simulate(until=0.1)
        >>> float(trajectory.get_values("gyroscope.gamma'")[0])
        10.0
    """

    field = 'physics/mechanics'
    topics = ('mechanics', 'ode', 'second order', 'rotation', 'gyroscope', 'lagrangian')
    references = ('Ryzhov et al. 2021',)
    until = 10.0
    dt = 0.01

    def __init__(
        self,
        spin: float = 50.0,
        beta: float = 0.3,
        *,
        A: float = 1.0,
        C: float = 2.0,
        A1: float = 0.5,
        B1: float = 0.5,
        C1: float = 0.5,
        A2: float = 1.0,
        torque: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.spin = spin
        self.beta = beta
        self.A = A
        self.C = C
        self.A1 = A1
        self.B1 = B1
        self.C1 = C1
        self.A2 = A2
        self.torque = torque

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a gyroscope in a gimbal, its rotor spinning.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        gyroscope = GyroscopeElement(
            'gyroscope',
            beta=self.beta,
            A=self.A,
            C=self.C,
            A1=self.A1,
            B1=self.B1,
            C1=self.C1,
            A2=self.A2,
            values={"gamma'": self.spin},
        )
        motion = GyroscopeInteraction('motion', target='gyroscope', torque=self.torque)
        self.system_ = System(self.label, elements=[gyroscope], interactions=[motion])
