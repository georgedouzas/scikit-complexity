"""Regulation models: plants with bounded control inputs that feedback controllers keep at a set point."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
from sage.all__sagemath_symbolics import cos, sin

from ....core.analysis import linearize, lqr
from ....core.base import Domain, Expression, parameter, state
from ....core.components import Element
from ....core.laws import Equation, Law, Rate, Symbols, derivative
from ....core.runtime import register_model
from ....core.solvers import RK4
from ....core.system import Observable, System
from ..realizations import PID, Integrator, StateFeedback

GRAVITY = 9.81
FORCE_LIMIT = 50.0


def _tilt(s: Symbols) -> Expression:
    """Return the angle of the pendulum from upright."""
    tilt: Expression = s.cart.theta
    return tilt


def _oscillator_energy(s: Symbols) -> Expression:
    """Return the energy per unit mass of the oscillator, `(x'^2 + omega^2 x^2) / 2`."""
    oscillator = s.oscillator
    energy: Expression = (oscillator.x_d1**2 + oscillator.omega**2 * oscillator.x**2) / 2
    return energy


class CartPendulum(Element):
    """Inverted pendulum on a cart that a bounded horizontal force pushes, after Åström and Murray.

    The cart of mass `M` at position `x` carries a point mass `m` at the end of a rod of length `l`, at the angle
    `theta` from upright, positive towards negative `x`. The equations of motion are
    `(M + m) x'' - m l theta'' cos(theta) + m l theta'^2 sin(theta) = force` and
    `l theta'' - x'' cos(theta) = g sin(theta)`, so upright is unstable and a positive force tilts the pendulum back
    from a positive angle.

    Args:
        label:
            The label of the cart.

        x:
            The initial position of the cart.

        theta:
            The initial angle from upright.

        force:
            The horizontal force on the cart, the control input.

        M:
            The mass of the cart.

        m:
            The mass at the end of the rod.

        l:
            The length of the rod.

        g:
            The acceleration of gravity.

    Examples:
        >>> from skcomplex.models.engineering.control import CartPendulum
        >>> CartPendulum('cart', theta=0.1).theta
        0.1
    """

    x: float = state(default=0.0, order=2)
    theta: float = state(default=0.0, order=2)
    force: float = parameter(default=0.0, adaptive=True, bounds=(-FORCE_LIMIT, FORCE_LIMIT))
    M: float = parameter(default=1.0, domain=Domain.POSITIVE)
    m: float = parameter(default=0.1, domain=Domain.POSITIVE)
    l: float = parameter(default=0.5, domain=Domain.POSITIVE)
    g: float = parameter(default=GRAVITY, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the equations of motion of the cart and the pendulum.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x_acceleration, theta_acceleration = derivative(s.x, 2), derivative(s.theta, 2)
        spin = derivative(s.theta, 1)
        return [
            Equation(
                (s.M + s.m) * x_acceleration
                - s.m * s.l * theta_acceleration * cos(s.theta)
                + s.m * s.l * spin**2 * sin(s.theta)
                == s.force,
            ),
            Equation(s.l * theta_acceleration - x_acceleration * cos(s.theta) == s.g * sin(s.theta)),
        ]


class DrivenOscillator(Element):
    """Damped harmonic oscillator that a bounded force drives, `x'' = -omega^2 x - 2 zeta omega x' + force`.

    Args:
        label:
            The label of the oscillator.

        x:
            The initial displacement.

        force:
            The force per unit mass, the control input.

        omega:
            The natural frequency.

        zeta:
            The damping ratio.

    Examples:
        >>> from skcomplex.models.engineering.control import DrivenOscillator
        >>> DrivenOscillator('oscillator', x=1.0).omega
        1.0
    """

    x: float = state(default=0.0, order=2)
    force: float = parameter(default=0.0, adaptive=True, bounds=(-FORCE_LIMIT, FORCE_LIMIT))
    omega: float = parameter(default=1.0, domain=Domain.POSITIVE)
    zeta: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate of the velocity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.x, -(s.omega**2) * s.x - 2 * s.zeta * s.omega * derivative(s.x, 1) + s.force)]


@register_model(
    field='engineering/control',
    tags=('control', 'feedback', 'pid', 'unstable equilibrium', 'ode'),
    references=('Åström and Murray 2008',),
    defaults={'until': 5.0, 'dt': 0.05, 'solver': RK4(dt=0.001)},
    observables=lambda system: [Observable('tilt', _tilt)],
)
def pid_inverted_pendulum(
    theta: float = 0.1,
    *,
    kp: float = 40.0,
    ki: float = 1.0,
    kd: float = 8.0,
    M: float = 1.0,
    m: float = 0.1,
    l: float = 0.5,
) -> System:
    """Return an inverted pendulum on a cart that a PID controller of its angle keeps upright.

    The controller sets the force on the cart to `kp e + ki ∫ e dt - kd theta'` from the error `e = -theta` of the
    angle, within the bounds of the force. Near upright the closed loop is stable when `kp > (M + m) g` and `kd > 0`,
    and the angle returns to upright while the cart drifts.

    Args:
        theta:
            The initial angle from upright.

        kp:
            The proportional gain.

        ki:
            The integral gain.

        kd:
            The derivative gain.

        M:
            The mass of the cart.

        m:
            The mass at the end of the rod.

        l:
            The length of the rod.

    Returns:
        system:
            The model, with the angle `cart.theta`, the position `cart.x`, the force `cart.force`, and the observable
            `tilt`.

    Examples:
        >>> from skcomplex.models.engineering.control import pid_inverted_pendulum
        >>> pid_inverted_pendulum(kp=50.0).interactions[0].kp
        50.0
    """
    cart = CartPendulum('cart', theta=theta, M=M, m=m, l=l)
    controller = PID(
        'pid',
        plant='cart',
        integrator='memory',
        control='force',
        measure='theta',
        kp=kp,
        ki=ki,
        kd=kd,
        limits=(-FORCE_LIMIT, FORCE_LIMIT),
    )
    return System(components=[cart, Integrator('memory')], interactions=[controller])


@register_model(
    field='engineering/control',
    tags=('control', 'feedback', 'linear-quadratic regulator', 'oscillator', 'ode'),
    references=('Åström and Murray 2008',),
    defaults={'until': 10.0, 'dt': 0.05, 'solver': RK4(dt=0.001)},
    observables=lambda system: [Observable('energy', _oscillator_energy)],
)
def lqr_oscillator(
    x: float = 1.0,
    *,
    omega: float = 1.0,
    zeta: float = 0.0,
    q: float = 1.0,
    r: float = 0.1,
) -> System:
    """Return a harmonic oscillator that a linear-quadratic regulator brings to rest.

    The gain `K` of the regulator minimises `∫ (q (x^2 + x'^2) + r force^2) dt` for the linearisation of the
    oscillator, which `lqr` computes from the Riccati equation, and the state feedback sets `force = -K (x, x')`. A
    smaller weight `r` of the force gives a larger gain and a faster return to rest.

    Args:
        x:
            The initial displacement.

        omega:
            The natural frequency.

        zeta:
            The damping ratio without control.

        q:
            The weight of the displacement and the velocity in the cost.

        r:
            The weight of the force in the cost.

    Returns:
        system:
            The model, with the displacement `oscillator.x`, the force `oscillator.force`, and the observable
            `energy`.

    Examples:
        >>> from skcomplex.models.engineering.control import lqr_oscillator
        >>> [round(value, 3) for value in lqr_oscillator().interactions[0].gain]
        [2.317, 3.825]
    """
    oscillator = DrivenOscillator('oscillator', x=x, omega=omega, zeta=zeta)
    model = linearize(System(components=[oscillator]), controls=['oscillator.force'])
    gain = lqr(model.A, model.B, q * np.eye(len(model.states)), [[r]])
    states = tuple(name.removeprefix('oscillator.') for name in model.states)
    regulator = StateFeedback(
        'regulator',
        plant='oscillator',
        control='force',
        states=states,
        gain=tuple(float(value) for value in gain[0]),
        limits=(-FORCE_LIMIT, FORCE_LIMIT),
    )
    return System(components=[oscillator], interactions=[regulator])
