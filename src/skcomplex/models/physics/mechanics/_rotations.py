"""Rotation models: a rotating swing, balls on rotating rods, and a gyroscope in a gimbal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, EquationLaw, RateLaw
from ....core.laws.utils import derivative
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class RotatingSwingElement(Element):
    """Rotating swing element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[object, StateSpec(period=2 * pi, order=2)] = 0.0,
        phi_d1: float = 0.0,
        theta: Annotated[object, StateSpec(order=2)] = 0.1,
        theta_d1: float = 0.0,
        J: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        m: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n_chairs: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 8.0,
        a: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        l: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        g: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 9.81,
        torque: Annotated[object, ParameterSpec()] = 0.0,
        friction: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the rotating swing element."""
        return [
            EquationLaw(
                (s.J + s.n_chairs * s.m * (s.a + s.l * sin(s.theta)) ** 2) * derivative(s.phi, 2)
                + 2
                * s.n_chairs
                * s.m
                * s.l
                * derivative(s.phi, 1)
                * derivative(s.theta, 1)
                * (s.a + s.l * sin(s.theta))
                * cos(s.theta)
                == s.torque - s.friction * derivative(s.phi, 1),
            ),
            EquationLaw(
                s.l * derivative(s.theta, 2)
                - derivative(s.phi, 1) ** 2 * (s.a + s.l * sin(s.theta)) * cos(s.theta)
                + s.g * sin(s.theta)
                == 0,
            ),
        ]


class RotatingBallElement(Element):
    """Rotating ball element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(period=2 * pi)] = 0.0,
        m: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        r: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        beta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = math.pi / 2,
        H: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.m = m
        self.r = r
        self.beta = beta
        self.H = H

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the rotating ball element."""
        return [RateLaw(s.theta, s.H / (s.m * s.r**2 * sin(s.beta) ** 2))]


class GyroscopeElement(Element):
    """Gyroscope element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alpha: Annotated[object, StateSpec(period=2 * pi, order=2)] = 0.0,
        alpha_d1: float = 0.0,
        beta: Annotated[object, StateSpec(order=2)] = 0.3,
        beta_d1: float = 0.0,
        gamma: Annotated[object, StateSpec(period=2 * pi, order=2)] = 0.0,
        gamma_d1: float = 0.0,
        A: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        A1: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        B1: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        C1: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        A2: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        torque: Annotated[object, ParameterSpec()] = 1.0,
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the gyroscope element."""
        return [
            EquationLaw(
                (s.A2 + (s.A + s.A1) * cos(s.beta) ** 2 + (s.C + s.C1) * sin(s.beta) ** 2) * derivative(s.alpha, 2)
                + 2
                * (s.C + s.C1 - s.A - s.A1)
                * sin(s.beta)
                * cos(s.beta)
                * derivative(s.beta, 1)
                * derivative(s.alpha, 1)
                + s.C * sin(s.beta) * derivative(s.gamma, 2)
                + s.C * cos(s.beta) * derivative(s.beta, 1) * derivative(s.gamma, 1)
                == 0,
            ),
            EquationLaw(
                (s.A + s.B1) * derivative(s.beta, 2)
                + (s.A + s.A1 - s.C - s.C1) * derivative(s.alpha, 1) ** 2 * cos(s.beta) * sin(s.beta)
                - s.C * derivative(s.alpha, 1) * derivative(s.gamma, 1) * cos(s.beta)
                == s.torque,
            ),
            EquationLaw(
                s.C
                * (
                    derivative(s.alpha, 2) * sin(s.beta)
                    + derivative(s.alpha, 1) * derivative(s.beta, 1) * cos(s.beta)
                    + derivative(s.gamma, 2)
                )
                == 0,
            ),
        ]


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'rotation', 'lagrangian'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 20.0, 'dt': 0.05},
)
def rotating_swing(
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
) -> System:
    """Return a rotating swing: chairs on chains hanging from the rods of a stand that turns about a vertical axis.

    The stand of inertia `J` turns by the angle `φ`, and the chairs, all deflected by the same angle `θ`, follow the
    Lagrange equations `[J + n m (a + l sin θ)²] φ'' + 2 n m l φ' θ' (a + l sin θ) cos θ = M - c φ'` and
    `l θ'' - φ'² (a + l sin θ) cos θ + g sin θ = 0`. At a steady speed `ω` the chairs settle at the angle where
    `g tan θ = ω² (a + l sin θ)`. Without torque and friction the angular momentum of the swing is conserved.

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

    Returns:
        system:
            The model, with the angle of the stand `swing.phi` and of the chains `swing.theta`, both of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import rotating_swing
        >>> rotating_swing(omega=1.0).elements[0].phi_d1
        1.0
    """
    return System(
        elements=[
            RotatingSwingElement(
                'swing',
                phi_d1=omega,
                theta=theta,
                n_chairs=n_chairs,
                J=J,
                m=m,
                a=a,
                l=l,
                g=g,
                torque=torque,
                friction=friction,
            ),
        ],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'rotation', 'angular momentum'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 10.0, 'dt': 0.02},
)
def rotating_balls_on_rods(
    omega: float = 1.0,
    *,
    r: float = 1.0,
    beta: float = math.pi / 2,
    m: float = 1.0,
) -> System:
    """Return two balls on rods that turn about a vertical axis and conserve their angular momentum.

    Each ball sits at the distance `r` along a rod fixed at the angle `β` to the axis, so its distance from the axis is
    `r sin β`. Its angular momentum `H = m ω r² sin² β` is fixed by the initial angular velocity, and the balls turn at
    `θ' = H / (m r² sin² β)`, so changing `r` during a run changes the angular velocity by `1 / r²`.

    Args:
        omega:
            The initial angular velocity.

        r:
            The distance of the balls from the joint of the rods.

        beta:
            The angle of the rods to the vertical axis, in radians.

        m:
            The mass of every ball.

    Returns:
        system:
            The model, with the angles `right.theta` and `left.theta` of the two balls.

    Examples:
        >>> from skcomplex.models.physics.mechanics import rotating_balls_on_rods
        >>> rotating_balls_on_rods(omega=2.0).elements[0].H
        2.0
    """
    momentum = m * omega * r**2 * math.sin(beta) ** 2
    return System(
        elements=[
            RotatingBallElement('right', theta=0.0, m=m, r=r, beta=beta, H=momentum),
            RotatingBallElement('left', theta=math.pi, m=m, r=r, beta=beta, H=momentum),
        ],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'rotation', 'gyroscope', 'lagrangian'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 10.0, 'dt': 0.01},
)
def gimballed_gyroscope(
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
) -> System:
    """Return a gyroscope in a gimbal: a rotor in an inner ring that turns in an outer ring about a fixed axis.

    The outer ring turns by `alpha` about the fixed axis, the inner ring by `beta` in the outer ring, and the rotor
    spins by `gamma` in the inner ring. The equations are the Lagrange equations of the kinetic energy
    `[A2 + (A + A1) cos² beta + (C + C1) sin² beta] alpha'² / 2 + (A + B1) beta'² / 2 + C alpha' gamma' sin beta
    + C gamma'² / 2`, with a torque on the axis of the inner ring, which makes the fast rotor precess about the fixed
    axis. The momenta conjugate to `alpha` and `gamma` are conserved.

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

    Returns:
        system:
            The model, with the angles `gyroscope.alpha`, `gyroscope.beta`, and `gyroscope.gamma`, all of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import gimballed_gyroscope
        >>> gimballed_gyroscope(spin=10.0).elements[0].gamma_d1
        10.0
    """
    return System(
        elements=[
            GyroscopeElement(
                'gyroscope',
                beta=beta,
                gamma_d1=spin,
                A=A,
                C=C,
                A1=A1,
                B1=B1,
                C1=C1,
                A2=A2,
                torque=torque,
            ),
        ],
    )
