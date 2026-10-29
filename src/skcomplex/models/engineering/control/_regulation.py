"""Regulation models: plants with bounded control inputs that feedback controllers keep at a set point."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo, sin
from sage.sets.real_set import RealSet

from ....analysis import LinearizationAnalysis
from ....core.elements import Element
from ....core.laws import BaseLaw, EquationLaw, RateLaw
from ....core.laws.utils import derivative
from ....core.system import System
from ....engine.solvers import RK4Solver
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._integrator import IntegratorElement
from ._pid import PIDInteraction
from ._state_feedback import StateFeedbackInteraction

GRAVITY = 9.81
FORCE_LIMIT = 50.0


def _tilt(s: SimpleNamespace) -> Expression:
    """Return the angle of the pendulum from upright."""
    tilt: Expression = s.cart.theta
    return tilt


def _oscillator_energy(s: SimpleNamespace) -> Expression:
    """Return the energy per unit mass of the oscillator, `(x'^2 + omega^2 x^2) / 2`."""
    oscillator = s.oscillator
    energy: Expression = (oscillator.x_d1**2 + oscillator.omega**2 * oscillator.x**2) / 2
    return energy


class CartPendulumElement(Element):
    """Inverted pendulum on a cart that a bounded horizontal force pushes, after Åström and Murray.

    The cart of mass `M` at position `x` carries a point mass `m` at the end of a rod of length `l`, at the angle
    `theta` from upright, positive towards negative `x`. The equations of motion are
    `(M + m) x'' - m l theta'' cos(theta) + m l theta'^2 sin(theta) = force` and
    `l theta'' - x'' cos(theta) = g sin(theta)`, so upright is unstable and a positive force tilts the pendulum back
    from a positive angle.

    Args:
        label:
            The label of the cart.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position of the cart.

        x_d1:
            The initial velocity of the cart.

        theta:
            The initial angle from upright.

        theta_d1:
            The initial angular velocity.

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
        >>> from skcomplex.models.engineering.control import CartPendulumElement
        >>> CartPendulumElement('cart', theta=0.1).theta
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        theta: Annotated[float, StateSpec(order=2)] = 0.0,
        theta_d1: float = 0.0,
        force: Annotated[float, ParameterSpec(adaptive=True, bounds=(-FORCE_LIMIT, FORCE_LIMIT))] = 0.0,
        M: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        l: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = GRAVITY,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.theta = theta
        self.theta_d1 = theta_d1
        self.force = force
        self.M = M
        self.m = m
        self.l = l
        self.g = g

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
            EquationLaw(
                (s.M + s.m) * x_acceleration
                - s.m * s.l * theta_acceleration * cos(s.theta)
                + s.m * s.l * spin**2 * sin(s.theta)
                == s.force,
            ),
            EquationLaw(s.l * theta_acceleration - x_acceleration * cos(s.theta) == s.g * sin(s.theta)),
        ]


class DrivenOscillatorElement(Element):
    """Damped harmonic oscillator that a bounded force drives, `x'' = -omega^2 x - 2 zeta omega x' + force`.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        force:
            The force per unit mass, the control input.

        omega:
            The natural frequency.

        zeta:
            The damping ratio.

    Examples:
        >>> from skcomplex.models.engineering.control import DrivenOscillatorElement
        >>> DrivenOscillatorElement('oscillator', x=1.0).omega
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        force: Annotated[float, ParameterSpec(adaptive=True, bounds=(-FORCE_LIMIT, FORCE_LIMIT))] = 0.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        zeta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.force = force
        self.omega = omega
        self.zeta = zeta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the velocity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.x, -(s.omega**2) * s.x - 2 * s.zeta * s.omega * derivative(s.x, 1) + s.force)]


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
