"""Regulation models: plants with bounded control inputs that feedback controllers keep at a set point."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....analysis import LinearizationAnalysis
from ....core.system import System
from ....engine.solvers import RK4Solver
from ....errors import ValidationError
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._cart_pendulum import FORCE_LIMIT, CartPendulumElement
from ._driven_oscillator import DrivenOscillatorElement
from ._integrator import IntegratorElement
from ._pid import PIDInteraction
from ._state_feedback import StateFeedbackInteraction


def _tilt(s: SimpleNamespace) -> Expression:
    """Return the angle of the pendulum from upright."""
    tilt: Expression = s.cart.theta
    return tilt


def _oscillator_energy(s: SimpleNamespace) -> Expression:
    """Return the energy per unit mass of the oscillator, `(x'^2 + omega^2 x^2) / 2`."""
    oscillator = s.oscillator
    energy: Expression = (oscillator.x_d1**2 + oscillator.omega**2 * oscillator.x**2) / 2
    return energy


@register_model(
    field='engineering/control',
    tags=('control', 'feedback', 'pid', 'unstable equilibrium', 'ode'),
    references=('Åström and Murray 2008',),
    defaults={'until': 5.0, 'dt': 0.05, 'solver': RK4Solver(dt=0.001)},
    observables=lambda system: [ObservableSpec('tilt', _tilt)],
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
    cart = CartPendulumElement('cart', theta=theta, M=M, m=m, l=l)
    controller = PIDInteraction(
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
    return System(elements=[cart, IntegratorElement('memory')], interactions=[controller])


@register_model(
    field='engineering/control',
    tags=('control', 'feedback', 'linear-quadratic regulator', 'oscillator', 'ode'),
    references=('Åström and Murray 2008',),
    defaults={'until': 10.0, 'dt': 0.05, 'solver': RK4Solver(dt=0.001)},
    observables=lambda system: [ObservableSpec('energy', _oscillator_energy)],
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
    oscillator, which `LinearizationAnalysis` computes from the Riccati equation, and the state feedback sets
    `force = -K (x, x')`. A smaller weight `r` of the force gives a larger gain and a faster return to rest.

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
    oscillator = DrivenOscillatorElement('oscillator', x=x, omega=omega, zeta=zeta)
    model = LinearizationAnalysis(['oscillator.force'], state_weights=q * np.eye(2), control_weights=[[r]])
    model.check().prepare(System(elements=[oscillator])).run()
    gain = model.gain_
    if gain is None:
        error_msg = 'The linearization of the oscillator gives no gain, since it has no cost weights.'
        raise ValidationError(error_msg)
    states = tuple(name.removeprefix('oscillator.') for name in model.states_)
    regulator = StateFeedbackInteraction(
        'regulator',
        plant='oscillator',
        control='force',
        states=states,
        gain=tuple(float(value) for value in gain[0]),
        limits=(-FORCE_LIMIT, FORCE_LIMIT),
    )
    return System(elements=[oscillator], interactions=[regulator])
