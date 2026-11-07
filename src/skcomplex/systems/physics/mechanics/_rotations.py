"""Rotation systems: a rotating swing, balls on rotating rods, and a gyroscope in a gimbal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import EulerLagrangeInteraction, RateInteraction
from ....core.components.laws.utils import derivative
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class GyroscopeElement(Element):
    """Gyroscope in a gimbal: a rotor in an inner ring that turns in an outer ring about a fixed axis.

    The angle `alpha` of the outer ring, `beta` of the inner ring, and `gamma` of the rotor have order 2. The motion
    comes from `GyroscopeInteraction`.

    Args:
        label:
            The label of the gyroscope.

        visual:
            How the gyroscope is drawn, or `None` for none.

        alpha:
            The initial angle of the outer ring.

        alpha_d1:
            The initial angular velocity of the outer ring.

        beta:
            The initial angle of the inner ring.

        beta_d1:
            The initial angular velocity of the inner ring.

        gamma:
            The initial angle of the rotor.

        gamma_d1:
            The initial angular velocity of the rotor.

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

    Examples:
        >>> from skcomplex.systems.physics.mechanics import GyroscopeElement
        >>> GyroscopeElement('gyroscope', gamma_d1=10.0).gamma_d1
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alpha: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.0,
        alpha_d1: float = 0.0,
        beta: Annotated[float, StateSpec(order=2)] = 0.3,
        beta_d1: float = 0.0,
        gamma: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.0,
        gamma_d1: float = 0.0,
        A: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        C: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        A1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        B1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        C1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        A2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        torque: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alpha = alpha
        self.alpha_d1 = alpha_d1
        self.beta = beta
        self.beta_d1 = beta_d1
        self.gamma = gamma
        self.gamma_d1 = gamma_d1
        self.A = A
        self.C = C
        self.A1 = A1
        self.B1 = B1
        self.C1 = C1
        self.A2 = A2
        self.torque = torque


class RotatingBallElement(Element):
    """Ball of mass `m` on a rod at the distance `r` from the joint, turning about a vertical axis by the angle `theta`.

    The rod is fixed at the angle `beta` to the axis, so the ball is at the distance `r sin β` from it, and `H` is its
    angular momentum about the axis.

    Args:
        label:
            The label of the ball.

        visual:
            How the ball is drawn, or `None` for none.

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
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = math.pi / 2,
        H: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.m = m
        self.r = r
        self.beta = beta
        self.H = H


class RotatingSwingElement(Element):
    """Rotating swing: chairs on chains hanging from the rods of a stand that turns about a vertical axis.

    The angle `phi` of the stand and the angle `theta` of the chains from the vertical have order 2. The motion comes
    from `RotatingSwingInteraction`.

    Args:
        label:
            The label of the swing.

        visual:
            How the swing is drawn, or `None` for none.

        phi:
            The initial angle of the stand.

        phi_d1:
            The initial angular velocity of the stand.

        theta:
            The initial angle of the chains.

        theta_d1:
            The initial angular velocity of the chains.

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

        torque:
            The torque that drives the stand.

        friction:
            The friction coefficient of the stand's axis.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingSwingElement
        >>> RotatingSwingElement('swing', n_chairs=4.0).n_chairs
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.0,
        phi_d1: float = 0.0,
        theta: Annotated[float, StateSpec(order=2)] = 0.1,
        theta_d1: float = 0.0,
        J: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n_chairs: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 8.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        torque: Annotated[float, ParameterSpec()] = 0.0,
        friction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.phi_d1 = phi_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.J = J
        self.m = m
        self.n_chairs = n_chairs
        self.a = a
        self.l = l
        self.g = g
        self.torque = torque
        self.friction = friction


class GyroscopeInteraction(EulerLagrangeInteraction):
    """Motion of a gimballed gyroscope, the Euler-Lagrange equations of its kinetic energy with a torque on `beta`.

    The Lagrangian is the kinetic energy `[A2 + (A + A1) cos² β + (C + C1) sin² β] alpha'² / 2 + (A + B1) β'² / 2
    + C alpha' gamma' sin β + C gamma'² / 2`, and the torque of the target acts on the axis of the inner ring.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the gyroscope.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import GyroscopeInteraction
        >>> GyroscopeInteraction('motion', target='gyroscope').states
        ('alpha', 'beta', 'gamma')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, states=('alpha', 'beta', 'gamma'), forcing=self._get_torques)

    @staticmethod
    def _get_torques(namespace: SimpleNamespace) -> list[Expression]:
        """Return the generalized forces, the torque on the inner ring and none on the outer ring and the rotor."""
        return [0, namespace.target.torque, 0]

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the kinetic energy of the rings and the rotor.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        da, db, dc = derivative(p.alpha, 1), derivative(p.beta, 1), derivative(p.gamma, 1)
        outer = p.A2 + (p.A + p.A1) * cos(p.beta) ** 2 + (p.C + p.C1) * sin(p.beta) ** 2
        return outer * da**2 / 2 + (p.A + p.B1) * db**2 / 2 + p.C * da * dc * sin(p.beta) + p.C * dc**2 / 2


class RotatingSwingInteraction(EulerLagrangeInteraction):
    """Motion of a rotating swing, the Euler-Lagrange equations with a torque and friction on the stand.

    The Lagrangian is `[J + n m (a + l sin θ)²] φ'² / 2 + n m l² θ'² / 2 + n m g l cos θ`, the torque drives the angle
    `φ` of the stand, and the Rayleigh dissipation `c φ'² / 2` adds the friction of its axis.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the swing.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingSwingInteraction
        >>> RotatingSwingInteraction('motion', target='swing').states
        ('phi', 'theta')
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            states=('phi', 'theta'),
            dissipation=self._get_friction,
            forcing=self._get_torques,
        )

    @staticmethod
    def _get_friction(namespace: SimpleNamespace) -> Expression:
        """Return the Rayleigh dissipation `c φ'² / 2` of the stand's axis."""
        return namespace.target.friction * derivative(namespace.target.phi, 1) ** 2 / 2

    @staticmethod
    def _get_torques(namespace: SimpleNamespace) -> list[Expression]:
        """Return the generalized forces, the torque on the stand and none on the chains."""
        return [namespace.target.torque, 0]

    def _get_lagrangian(self, namespace: SimpleNamespace) -> Expression:
        """Return the Lagrangian of the stand and the chairs.

        Args:
            namespace:
                The namespace of the symbols the Lagrangian can use.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        p = namespace.target
        chairs = p.n_chairs * p.m
        radius = p.a + p.l * sin(p.theta)
        kinetic = (p.J + chairs * radius**2) * derivative(p.phi, 1) ** 2 / 2
        kinetic += chairs * p.l**2 * derivative(p.theta, 1) ** 2 / 2
        return kinetic + chairs * p.g * p.l * cos(p.theta)


class RotatingSwingSystem(BaseSystem):
    """A rotating swing: chairs on chains hanging from the rods of a stand that turns about a vertical axis.

    The stand of inertia `J` turns by the angle `φ`, and the chairs, all deflected by the same angle `θ`, follow the
    Lagrange equations `[J + n m (a + l sin θ)²] φ'' + 2 n m l φ' θ' (a + l sin θ) cos θ = M - c φ'` and `l θ'' - φ'² (a
    + l sin θ) cos θ + g sin θ = 0`. At a steady speed `ω` the chairs settle at the angle where `g tan θ = ω² (a + l sin
    θ)`. Without torque and friction the angular momentum of the swing is conserved. The system has the angle of the
    stand `swing.phi` and of the chains `swing.theta`, both of order 2.

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

    Examples:
        >>> from skcomplex.systems.physics.mechanics import RotatingSwingSystem
        >>> RotatingSwingSystem(omega=1.0).prepare().system_.elements[0].phi_d1
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
    ) -> None:
        super().__init__(label)
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

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        return System(
            self.label,
            elements=[
                RotatingSwingElement(
                    'swing',
                    phi_d1=self.omega,
                    theta=self.theta,
                    n_chairs=self.n_chairs,
                    J=self.J,
                    m=self.m,
                    a=self.a,
                    l=self.l,
                    g=self.g,
                    torque=self.torque,
                    friction=self.friction,
                ),
            ],
            interactions=[RotatingSwingInteraction('motion', target='swing')],
        )


class RotatingBallsOnRodsSystem(BaseSystem):
    """Two balls on rods that turn about a vertical axis and conserve their angular momentum.

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
    ) -> None:
        super().__init__(label)
        self.omega = omega
        self.r = r
        self.beta = beta
        self.m = m

    def _build_system(self) -> System:
        """Return two balls on rods that turn about a vertical axis and conserve their angular momentum."""
        momentum = self.m * self.omega * self.r**2 * math.sin(self.beta) ** 2
        return System(
            self.label,
            elements=[
                RotatingBallElement('right', theta=0.0, m=self.m, r=self.r, beta=self.beta, H=momentum),
                RotatingBallElement('left', theta=math.pi, m=self.m, r=self.r, beta=self.beta, H=momentum),
            ],
            interactions=[
                RateInteraction(
                    f'{ball}_turn',
                    target=ball,
                    state='theta',
                    rate=lambda s: s.target.H / (s.target.m * s.target.r**2 * sin(s.target.beta) ** 2),
                )
                for ball in ('right', 'left')
            ],
        )


class GimballedGyroscopeSystem(BaseSystem):
    """A gyroscope in a gimbal: a rotor in an inner ring that turns in an outer ring about a fixed axis.

    The outer ring turns by `alpha` about the fixed axis, the inner ring by `beta` in the outer ring, and the rotor
    spins by `gamma` in the inner ring. The equations are the Lagrange equations of the kinetic energy `[A2 + (A + A1)
    cos² beta + (C + C1) sin² beta] alpha'² / 2 + (A + B1) beta'² / 2 + C alpha' gamma' sin beta + C gamma'² / 2`, with
    a torque on the axis of the inner ring, which makes the fast rotor precess about the fixed axis. The momenta
    conjugate to `alpha` and `gamma` are conserved. The system has the angles `gyroscope.alpha`, `gyroscope.beta`, and
    `gyroscope.gamma`, all of order 2.

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

    Examples:
        >>> from skcomplex.systems.physics.mechanics import GimballedGyroscopeSystem
        >>> GimballedGyroscopeSystem(spin=10.0).prepare().system_.elements[0].gamma_d1
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
    ) -> None:
        super().__init__(label)
        self.spin = spin
        self.beta = beta
        self.A = A
        self.C = C
        self.A1 = A1
        self.B1 = B1
        self.C1 = C1
        self.A2 = A2
        self.torque = torque

    def _build_system(self) -> System:
        """Return a gyroscope in a gimbal: a rotor in an inner ring that turns in an outer ring about a fixed axis."""
        return System(
            self.label,
            elements=[
                GyroscopeElement(
                    'gyroscope',
                    beta=self.beta,
                    gamma_d1=self.spin,
                    A=self.A,
                    C=self.C,
                    A1=self.A1,
                    B1=self.B1,
                    C1=self.C1,
                    A2=self.A2,
                    torque=self.torque,
                ),
            ],
            interactions=[GyroscopeInteraction('motion', target='gyroscope')],
        )
