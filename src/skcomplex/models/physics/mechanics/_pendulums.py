"""Pendulum models: double, coupled, parametric, magnetic, and ballistic pendulums, carts, and clocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import cos, sign, sin, sqrt

from ....core.base import Domain, Expression, parameter, role, state
from ....core.model import (
    Element,
    Equation,
    EuclideanSpace,
    Event,
    HarmonicPotential,
    Interaction,
    LinearDrag,
    Particle,
    Potential,
    Rate,
    Symbols,
    System,
    derivative,
)
from ....core.runtime import register_model
from ._mechanics import Pendulum

DoublePendulum = Element.define(
    'DoublePendulum',
    states={
        'theta1': state(default=math.pi / 2, domain=Domain.ANGLE, order=2),
        'theta2': state(default=math.pi / 2, domain=Domain.ANGLE, order=2),
    },
    parameters={
        'm1': parameter(default=1.0, domain=Domain.POSITIVE),
        'm2': parameter(default=1.0, domain=Domain.POSITIVE),
        'l1': parameter(default=1.0, domain=Domain.POSITIVE),
        'l2': parameter(default=1.0, domain=Domain.POSITIVE),
        'g': parameter(default=9.81, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Equation(
            (s.m1 + s.m2) * s.l1 * derivative(s.theta1, 2)
            + s.m2 * s.l2 * derivative(s.theta2, 2) * cos(s.theta1 - s.theta2)
            + s.m2 * s.l2 * derivative(s.theta2, 1) ** 2 * sin(s.theta1 - s.theta2)
            + (s.m1 + s.m2) * s.g * sin(s.theta1)
            == 0,
        ),
        Equation(
            s.l2 * derivative(s.theta2, 2)
            + s.l1 * derivative(s.theta1, 2) * cos(s.theta1 - s.theta2)
            - s.l1 * derivative(s.theta1, 1) ** 2 * sin(s.theta1 - s.theta2)
            + s.g * sin(s.theta2)
            == 0,
        ),
    ],
)

PendulumSpring = Interaction.define(
    'PendulumSpring',
    roles=['first', 'second'],
    parameters={'k': parameter(default=1.0, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [
        Rate(
            s.first.theta,
            s.k
            * (s.second.l * sin(s.second.theta) - s.first.l * sin(s.first.theta))
            * cos(s.first.theta)
            / (s.first.m * s.first.l),
        ),
        Rate(
            s.second.theta,
            -s.k
            * (s.second.l * sin(s.second.theta) - s.first.l * sin(s.first.theta))
            * cos(s.second.theta)
            / (s.second.m * s.second.l),
        ),
    ],
)

CartPendulum = Element.define(
    'CartPendulum',
    states={'y': state(default=0.0, order=2), 'theta': state(default=0.5, domain=Domain.ANGLE, order=2)},
    parameters={
        'm1': parameter(default=1.0, domain=Domain.POSITIVE),
        'm2': parameter(default=1.0, domain=Domain.POSITIVE),
        'l': parameter(default=1.0, domain=Domain.POSITIVE),
        'g': parameter(default=9.81, domain=Domain.POSITIVE),
        'F': 0.0,
    },
    laws=lambda s: [
        Equation(
            (s.m1 + s.m2) * derivative(s.y, 2)
            + s.m2 * s.l * derivative(s.theta, 2) * cos(s.theta)
            - s.m2 * s.l * derivative(s.theta, 1) ** 2 * sin(s.theta)
            == s.F,
        ),
        Equation(s.l * derivative(s.theta, 2) + derivative(s.y, 2) * cos(s.theta) + s.g * sin(s.theta) == 0),
    ],
)

ParametricPendulum = Element.define(
    'ParametricPendulum',
    states={'theta': state(default=0.1, domain=Domain.ANGLE, order=2)},
    parameters={
        'g': parameter(default=9.81, domain=Domain.POSITIVE),
        'l': parameter(default=1.0, domain=Domain.POSITIVE),
        'epsilon': 0.2,
        'Omega': parameter(default=2 * math.sqrt(9.81), domain=Domain.NONNEGATIVE),
        'c': parameter(default=0.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Equation(
            derivative(s.theta, 2)
            + s.c * derivative(s.theta, 1)
            + s.g / s.l * (1 + s.epsilon * cos(s.Omega * s.t)) * sin(s.theta)
            == 0,
        ),
    ],
)

BallisticPendulum = Element.define(
    'BallisticPendulum',
    states={'x': state(default=-1.0, order=2), 'theta': state(default=0.0, domain=Domain.ANGLE, order=2)},
    parameters={
        'm': parameter(default=0.01, domain=Domain.POSITIVE),
        'M': parameter(default=1.0, domain=Domain.POSITIVE),
        'l': parameter(default=1.0, domain=Domain.POSITIVE),
        'g': parameter(default=9.81, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Rate(s.x, 0),
        Equation(s.l * derivative(s.theta, 2) == -s.g * sin(s.theta)),
        Event(
            s.x,
            {
                derivative(s.x, 1): 0,
                derivative(s.theta, 1): s.m * derivative(s.x, 1) / ((s.m + s.M) * s.l),
            },
            direction=1,
            label='impact',
        ),
    ],
)

ClockPendulum = Element.define(
    'ClockPendulum',
    states={'theta': state(default=0.1, order=2), 'x': state(default=0.0, order=2)},
    parameters={
        'm': parameter(default=1.0, domain=Domain.POSITIVE),
        'l': parameter(default=1.0, domain=Domain.POSITIVE),
        'M': parameter(default=1.0, domain=Domain.POSITIVE),
        'g': parameter(default=9.8, domain=Domain.POSITIVE),
        'rho': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'lam': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'theta_c': parameter(default=0.1, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Equation(
            s.m * s.l * derivative(s.theta, 2)
            + 2 * s.lam * sign(abs(s.theta) - s.theta_c) * derivative(s.theta, 1)
            + s.m * s.g * sin(s.theta)
            == -s.m * derivative(s.x, 2) * cos(s.theta),
        ),
        Equation(
            (s.M + s.m) * derivative(s.x, 2)
            + 2 * s.rho * derivative(s.x, 1)
            + s.m * s.l * derivative(s.theta, 2) * cos(s.theta)
            == s.m * s.l * derivative(s.theta, 1) ** 2 * sin(s.theta),
        ),
    ],
)


def _support_response(pendulum: Symbols, force: Expression) -> list[Rate]:
    """Return the accelerations of a supported pendulum and its support under a horizontal force on the support."""
    inertia = pendulum.M + pendulum.m * sin(pendulum.theta) ** 2
    return [
        Rate(pendulum.x, force / inertia),
        Rate(pendulum.theta, -cos(pendulum.theta) * force / (pendulum.l * inertia)),
    ]


SupportSpring = Interaction.define(
    'SupportSpring',
    roles=['first', 'second'],
    parameters={'k': parameter(default=1.0, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [
        *_support_response(s.first, s.k * (s.second.x - s.first.x)),
        *_support_response(s.second, -s.k * (s.second.x - s.first.x)),
    ],
)


class Magnet(Potential):
    """Magnet below the plane of a pendulum bob, which attracts it with energy `-gamma / sqrt(|x - center|**2 + d**2)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the bob.

        center:
            The position of the magnet in the plane.

        gamma:
            The strength of the magnet.

        d:
            The height of the bob above the plane of the magnets.

    Examples:
        >>> from skcomplex.models.physics.mechanics import Magnet
        >>> Magnet('magnet', target='bob', center=[1.0, 0.0]).gamma
        1.0
    """

    target: str = role()
    center: list[float] = parameter(default=0.0, vector=True)
    gamma: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    d: float = parameter(default=0.3, domain=Domain.POSITIVE)

    def energy(self, s: Symbols) -> Expression:
        """Return `-gamma / sqrt(|x - center|**2 + d**2)`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        displacement = s.target.x - s.center
        return -s.gamma / sqrt(displacement.dot_product(displacement) + s.d**2)


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'chaos', 'lagrangian'),
    references=('Ryzhov et al. 2021', 'Argun et al. 2021'),
    defaults={'until': 20.0, 'dt': 0.02},
)
def double_pendulum(
    theta1: float = math.pi / 2,
    theta2: float = math.pi / 2,
    omega1: float = 0.0,
    omega2: float = 0.0,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> System:
    """Return a double pendulum, a pendulum hanging from the bob of another, written as its two Lagrange equations.

    The equations `(m1 + m2) l1 θ1'' + m2 l2 θ2'' cos(θ1 - θ2) + m2 l2 θ2'² sin(θ1 - θ2) + (m1 + m2) g sin θ1 = 0`
    and `l2 θ2'' + l1 θ1'' cos(θ1 - θ2) - l1 θ1'² sin(θ1 - θ2) + g sin θ2 = 0` are solved jointly for the angular
    accelerations. The motion is chaotic at large amplitudes.

    Args:
        theta1:
            The initial angle of the upper rod from the vertical, in radians.

        theta2:
            The initial angle of the lower rod from the vertical, in radians.

        omega1:
            The initial angular velocity of the upper rod.

        omega2:
            The initial angular velocity of the lower rod.

        m1:
            The mass of the upper bob.

        m2:
            The mass of the lower bob.

        l1:
            The length of the upper rod.

        l2:
            The length of the lower rod.

        g:
            The gravitational acceleration.

    Returns:
        system:
            The model, with the angles `pendulum.theta1` and `pendulum.theta2` of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import double_pendulum
        >>> double_pendulum(m2=2.0).elements[0].m2
        2.0
    """
    return System(
        elements=[
            DoublePendulum(
                'pendulum',
                theta1=theta1,
                theta2=theta2,
                theta1_d1=omega1,
                theta2_d1=omega2,
                m1=m1,
                m2=m2,
                l1=l1,
                l2=l2,
                g=g,
            ),
        ],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'oscillation', 'normal modes'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 40.0, 'dt': 0.05},
)
def coupled_pendulums(
    theta1: float = 0.2,
    theta2: float = 0.0,
    *,
    k: float = 0.5,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> System:
    """Return two pendulums whose bobs are joined by a spring, which pass their energy back and forth.

    The spring stretches with the horizontal distance `l2 sin θ2 - l1 sin θ1` of the bobs and adds the torque
    `k l1 (l2 sin θ2 - l1 sin θ1) cos θ1` to the first pendulum and the opposite one to the second. Identical pendulums
    have the normal modes `sqrt(g / l)` in phase and `sqrt(g / l + 2 k / m)` in antiphase.

    Args:
        theta1:
            The initial angle of the first pendulum.

        theta2:
            The initial angle of the second pendulum.

        k:
            The stiffness of the spring.

        m1:
            The mass of the first bob.

        m2:
            The mass of the second bob.

        l1:
            The length of the first pendulum.

        l2:
            The length of the second pendulum.

        g:
            The gravitational acceleration.

    Returns:
        system:
            The model, with the pendulums `left` and `right` and their angles `left.theta` and `right.theta`.

    Examples:
        >>> from skcomplex.models.physics.mechanics import coupled_pendulums
        >>> coupled_pendulums(k=2.0).interactions[0].k
        2.0
    """
    return System(
        elements=[
            Pendulum('left', theta=theta1, m=m1, l=l1, g=g),
            Pendulum('right', theta=theta2, m=m2, l=l2, g=g),
        ],
        interactions=[PendulumSpring('spring', first='left', second='right', k=k)],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'lagrangian', 'momentum'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 10.0, 'dt': 0.02},
)
def cart_pendulum(
    theta: float = 0.5,
    y: float = 0.0,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l: float = 1.0,
    g: float = 9.81,
    F: float = 0.0,
) -> System:
    """Return a pendulum hanging from a cart that slides without friction along a horizontal track.

    The Lagrange equations `(m1 + m2) y'' + m2 l (θ'' cos θ - θ'² sin θ) = F` and `l θ'' + y'' cos θ + g sin θ = 0`
    couple the position `y` of the cart and the angle `θ` of the pendulum. Without a force the horizontal momentum is
    conserved, so the cart swings against the pendulum.

    Args:
        theta:
            The initial angle of the pendulum.

        y:
            The initial position of the cart.

        m1:
            The mass of the cart.

        m2:
            The mass of the bob.

        l:
            The length of the rod.

        g:
            The gravitational acceleration.

        F:
            A constant horizontal force on the cart.

    Returns:
        system:
            The model, with the cart position `cart.y` and the angle `cart.theta`, both of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import cart_pendulum
        >>> cart_pendulum(m1=5.0).elements[0].m1
        5.0
    """
    return System(elements=[CartPendulum('cart', y=y, theta=theta, m1=m1, m2=m2, l=l, g=g, F=F)])


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'forcing', 'parametric resonance'),
    references=('Dilão 2023',),
    defaults={'until': 30.0, 'dt': 0.02},
)
def parametric_pendulum(
    theta: float = 0.1,
    omega: float = 0.0,
    *,
    g: float = 9.81,
    l: float = 1.0,
    epsilon: float = 0.2,
    Omega: float = 2 * math.sqrt(9.81),
    c: float = 0.0,
) -> System:
    """Return a pendulum whose suspension point oscillates vertically, `θ'' + c θ' + (g / l)(1 + ε cos Ω t) sin θ = 0`.

    The vertical acceleration of the suspension point modulates gravity. Small oscillations grow when `Ω` is close to
    twice the natural frequency `sqrt(g / l)`, the parametric resonance of a swing.

    Args:
        theta:
            The initial angle.

        omega:
            The initial angular velocity.

        g:
            The gravitational acceleration.

        l:
            The length of the rod.

        epsilon:
            The amplitude of the acceleration of the suspension point, relative to `g`.

        Omega:
            The angular frequency of the suspension point.

        c:
            The friction coefficient.

    Returns:
        system:
            The model, with the angle `swing.theta` of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import parametric_pendulum
        >>> parametric_pendulum(epsilon=0.5).elements[0].epsilon
        0.5
    """
    return System(
        elements=[
            ParametricPendulum('swing', theta=theta, theta_d1=omega, g=g, l=l, epsilon=epsilon, Omega=Omega, c=c),
        ],
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'multistability', 'fractal basins'),
    references=('Datseris and Parlitz 2022',),
    defaults={'until': 50.0, 'dt': 0.05},
)
def magnetic_pendulum(
    x: float = 1.0,
    y: float = 1.5,
    *,
    n_magnets: int = 3,
    omega: float = 1.0,
    q: float = 0.2,
    gamma: float = 1.0,
    d: float = 0.3,
) -> System:
    """Return a magnetic bob on a string swinging above magnets on a circle, which comes to rest at one of them.

    The bob follows `x'' = -ω² x - q x' - Σ gamma (x - x_i) / d_i³` with `d_i = sqrt(|x - x_i|² + d²)`. The basins of
    the magnets have fractal boundaries.

    Args:
        x:
            The initial horizontal position of the bob.

        y:
            The initial vertical position of the bob.

        n_magnets:
            The number of magnets, evenly spaced on the unit circle.

        omega:
            The natural frequency of the pendulum.

        q:
            The friction coefficient.

        gamma:
            The strength of every magnet.

        d:
            The height of the bob above the plane of the magnets.

    Returns:
        system:
            The model, with the bob `bob` and its position `bob.x` in two dimensions.

    Examples:
        >>> from skcomplex.models.physics.mechanics import magnetic_pendulum
        >>> len(magnetic_pendulum(n_magnets=4).interactions)
        6
    """
    angles = 2 * np.pi * np.arange(n_magnets) / n_magnets
    magnets = [
        Magnet(
            f'magnet_{index}',
            target='bob',
            center=[round(float(np.cos(angle)), 12), round(float(np.sin(angle)), 12)],
            gamma=gamma,
            d=d,
        )
        for index, angle in enumerate(angles)
    ]
    return System(
        elements=[Particle('bob', x=[x, y], v=[0.0, 0.0])],
        interactions=[
            HarmonicPotential('string', target='bob', k=omega**2),
            LinearDrag('friction', target='bob', gamma=q),
            *magnets,
        ],
        space=EuclideanSpace(2),
    )


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'events', 'impacts', 'momentum', 'hybrid'),
    references=('Ryzhov et al. 2021',),
    defaults={'until': 3.0, 'dt': 0.01},
)
def ballistic_pendulum(
    speed: float = 200.0,
    distance: float = 1.0,
    *,
    m: float = 0.01,
    M: float = 1.0,
    l: float = 1.0,
    g: float = 9.81,
) -> System:
    """Return a bullet that sticks in the block of a pendulum at rest, which then swings up.

    The bullet flies horizontally towards the block, and the impact is an event. Momentum is conserved in the
    perfectly inelastic collision, so the pendulum starts with the speed `m v / (m + M)` and rises to the height
    `(m v / (m + M))² / (2 g)`. The bullet position stops at the block after the impact.

    Args:
        speed:
            The speed of the bullet.

        distance:
            The initial distance of the bullet from the block.

        m:
            The mass of the bullet.

        M:
            The mass of the block.

        l:
            The length of the rod.

        g:
            The gravitational acceleration.

    Returns:
        system:
            The model, with the bullet position `pendulum.x` and the angle `pendulum.theta`, both of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import ballistic_pendulum
        >>> ballistic_pendulum(distance=2.0).elements[0].x
        -2.0
    """
    return System(elements=[BallisticPendulum('pendulum', x=-distance, x_d1=speed, m=m, M=M, l=l, g=g)])


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'second order', 'synchronization', 'limit cycle'),
    references=('Huygens 1673', 'Dilão 2023', 'Dilão 2023'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def huygens_clocks(
    n_clocks: int = 2,
    *,
    theta: float = 0.1,
    k: float = 1.0,
    m: float = 1.0,
    l: float = 1.0,
    M: float = 1.0,
    g: float = 9.8,
    rho: float = 0.1,
    lam: float = 0.1,
    theta_c: float = 0.1,
) -> System:
    """Return pendulum clocks on supports joined by springs, which synchronize through the motion of the supports.

    Every pendulum hangs from a support of mass `M` that moves horizontally with damping `rho`, and neighbouring
    supports are joined by springs of stiffness `k`. A pendulum follows `m l θ'' + f(θ) θ' + m g sin θ = -m x'' cos θ`
    and its support `(M + m) x'' + 2 rho x' + m l θ'' cos θ = m l θ'² sin θ + spring forces`. The escapement
    `f(θ) = 2 λ sgn(|θ| - θ_c)` is `-2 λ` for `|θ| < θ_c`, pumping energy in, and `2 λ` beyond, so every clock has a
    limit cycle. With `λ = 0` the clocks are plain pendulums. Two identical clocks synchronize in antiphase.

    Args:
        n_clocks:
            The number of clocks in the row.

        theta:
            The initial angle of the first clock. The others start at rest.

        k:
            The stiffness of the springs between neighbouring supports.

        m:
            The mass of every bob.

        l:
            The length of every pendulum.

        M:
            The mass of every support.

        g:
            The gravitational acceleration.

        rho:
            The damping of the supports.

        lam:
            The strength of the escapement.

        theta_c:
            The angle below which the escapement pumps energy in.

    Returns:
        system:
            The model, with the angles `clock_<i>.theta` and the support positions `clock_<i>.x`, all of order 2.

    Examples:
        >>> from skcomplex.models.physics.mechanics import huygens_clocks
        >>> len(huygens_clocks(3).interactions)
        2
    """
    clocks = [
        ClockPendulum(
            f'clock_{index}',
            theta=theta if index == 0 else 0.0,
            m=m,
            l=l,
            M=M,
            g=g,
            rho=rho,
            lam=lam,
            theta_c=theta_c,
        )
        for index in range(n_clocks)
    ]
    springs = [
        SupportSpring(f'spring_{index}', first=f'clock_{index}', second=f'clock_{index + 1}', k=k)
        for index in range(n_clocks - 1)
    ]
    return System(elements=clocks, interactions=springs)
