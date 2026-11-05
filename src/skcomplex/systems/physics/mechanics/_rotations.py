"""Rotation systems: a rotating swing, balls on rotating rods, and a gyroscope in a gimbal."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from sage.all__sagemath_symbolics import sin

from ....core.components.interactions import RateInteraction
from ....core.components.system import System
from ..._catalog import register_system
from ._gyroscope import GyroscopeElement
from ._gyroscope_interaction import GyroscopeInteraction
from ._rotating_ball import RotatingBallElement
from ._rotating_swing import RotatingSwingElement
from ._rotating_swing_interaction import RotatingSwingInteraction


@register_system(
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
            The system, with the angle of the stand `swing.phi` and of the chains `swing.theta`, both of order 2.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import rotating_swing
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
        interactions=[RotatingSwingInteraction('motion', target='swing')],
    )


@register_system(
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
            The system, with the angles `right.theta` and `left.theta` of the two balls.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import rotating_balls_on_rods
        >>> rotating_balls_on_rods(omega=2.0).elements[0].H
        2.0
    """
    momentum = m * omega * r**2 * math.sin(beta) ** 2
    return System(
        elements=[
            RotatingBallElement('right', theta=0.0, m=m, r=r, beta=beta, H=momentum),
            RotatingBallElement('left', theta=math.pi, m=m, r=r, beta=beta, H=momentum),
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


@register_system(
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
            The system, with the angles `gyroscope.alpha`, `gyroscope.beta`, and `gyroscope.gamma`, all of order 2.

    Examples:
        >>> from skcomplex.systems.physics.mechanics import gimballed_gyroscope
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
        interactions=[GyroscopeInteraction('motion', target='gyroscope')],
    )
