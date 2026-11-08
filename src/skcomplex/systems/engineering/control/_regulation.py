"""Regulation systems: plants with bounded control inputs kept at a set point.

Feedback controllers keep the plants at the set point.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, oo, sin
from sage.sets.real_set import RealSet

from ....core.analysis import linearization
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, BoundaryLaw, EquationLaw, RateLaw
from ....core.components.laws.utils import clip, derivative
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import RK4Solver
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec

GRAVITY = 9.81


FORCE_LIMIT = 50.0


class CartPendulumElement(Element):
    """Inverted pendulum on a cart that a bounded horizontal force pushes.

    It follows Åström and Murray. The cart of mass `M` at position `x` carries a point mass `m` at the end of a rod of
    length `l`, at the angle `theta` from upright, positive towards negative `x`. The equations of motion are
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
        >>> from skcomplex.systems.engineering.control import CartPendulumElement
        >>> CartPendulumElement('cart', theta=0.1).theta
        0.1
    """

    law_classes = (EquationLaw,)

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
    """Damped harmonic oscillator that a bounded force drives.

    It follows `x'' = -omega^2 x - 2 zeta omega x' + force`.

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
        >>> from skcomplex.systems.engineering.control import DrivenOscillatorElement
        >>> DrivenOscillatorElement('oscillator', x=1.0).omega
        1.0
    """

    law_classes = (RateLaw,)

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


class IntegratorElement(Element):
    """Element that holds the integral of a signal, whose rate a controller sets.

    Args:
        label:
            The label of the integrator.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The initial value of the integral.

    Examples:
        >>> from skcomplex.systems.engineering.control import IntegratorElement
        >>> IntegratorElement('memory').value
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value


class PIDInteraction(Interaction):
    """Proportional-integral-derivative controller of a control input of a plant.

    It sets the control input from the error of a measure. The error is `e = setpoint - y` for the measured state `y`,
    and the controller sets the control input to `u = kp e + ki I - kd y'`, clipped to the limits, with the derivative
    taken on the measure. The integral `I' = e` of the error lives in an `IntegratorElement` bound to the role
    `integrator`. The control input is an adaptive parameter of the plant, which the controller sets after every solver
    step, so it holds within a step like a sampled controller. The derivative term reads the first derivative of the
    measure, so a measure with a derivative term is a state of order 2.

    Args:
        label:
            The label of the controller.

        plant:
            The path of the element that holds the measure and the control input.

        integrator:
            The path of the integrator of the error.

        kp:
            The proportional gain.

        ki:
            The integral gain.

        kd:
            The derivative gain.

        setpoint:
            The value the measure is kept at.

        control:
            The name of the adaptive parameter of the plant that the controller sets.

        measure:
            The name of the measured state of the plant.

        limits:
            The lower and upper limit of the control input, or `None` for no limits.

    Examples:
        >>> from skcomplex.systems.engineering.control import PIDInteraction
        >>> PIDInteraction('pid', plant='cart', integrator='memory', control='force', measure='theta', kp=40.0).kp
        40.0
    """

    law_classes = (RateLaw, BoundaryLaw)

    def __init__(
        self,
        label: str,
        *,
        plant: Annotated[str, RoleSpec()],
        integrator: Annotated[str, RoleSpec()],
        kp: Annotated[float, ParameterSpec()] = 1.0,
        ki: Annotated[float, ParameterSpec()] = 0.0,
        kd: Annotated[float, ParameterSpec()] = 0.0,
        setpoint: Annotated[float, ParameterSpec()] = 0.0,
        control: str = 'u',
        measure: str = 'y',
        limits: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(label)
        self.plant = plant
        self.integrator = integrator
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.control = control
        self.measure = measure
        self.limits = limits

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the integral of the error and the control input it sets.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        measured = getattr(s.plant, self.measure)
        error = s.setpoint - measured
        output = s.kp * error + s.ki * s.integrator.value
        if self.kd != 0:
            output -= s.kd * derivative(measured, 1)
        return [
            RateLaw(s.integrator.value, error),
            BoundaryLaw(getattr(s.plant, self.control), clip(output, *(self.limits or (None, None)))),
        ]


class StateFeedbackInteraction(Interaction):
    """Linear state feedback `u = -K x` of a control input of a plant.

    It sets the control input from the states of the plant. The control input is an adaptive parameter of the plant,
    which the controller sets after every solver step, so it holds within a step like a sampled controller. A gain from
    `linearization` of the plant makes the controller a linear-quadratic regulator.

    Args:
        label:
            The label of the controller.

        plant:
            The path of the element that holds the states and the control input.

        control:
            The name of the adaptive parameter of the plant that the controller sets.

        states:
            The names of the fed back states of the plant, such as `('x', 'x_d1')` for a position and its velocity.

        gain:
            The gain of every state, in the order of `states`.

        limits:
            The lower and upper limit of the control input, or `None` for no limits.

    Examples:
        >>> from skcomplex.systems.engineering.control import StateFeedbackInteraction
        >>> StateFeedbackInteraction(
        ...     'regulator', plant='mass', control='force', states=('x', 'x_d1'), gain=(1.0, 1.7)
        ... ).gain
        (1.0, 1.7)
    """

    law_classes = (BoundaryLaw,)

    def __init__(
        self,
        label: str,
        *,
        plant: Annotated[str, RoleSpec()],
        control: str = 'u',
        states: Sequence[str] = (),
        gain: Sequence[float] = (),
        limits: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(label)
        self.plant = plant
        self.control = control
        self.states = states
        self.gain = gain
        self.limits = limits

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the control input the feedback sets.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        feedback = -sum(
            (float(gain) * getattr(s.plant, name) for name, gain in zip(self.states, self.gain, strict=True)),
            0,
        )
        return [BoundaryLaw(getattr(s.plant, self.control), clip(feedback, *(self.limits or (None, None))))]


def _tilt(s: SimpleNamespace) -> Expression:
    """Return the angle of the pendulum from upright."""
    tilt: Expression = s.cart.theta
    return tilt


def _oscillator_energy(s: SimpleNamespace) -> Expression:
    """Return the energy per unit mass of the oscillator, `(x'^2 + omega^2 x^2) / 2`."""
    oscillator = s.oscillator
    energy: Expression = (oscillator.x_d1**2 + oscillator.omega**2 * oscillator.x**2) / 2
    return energy


class PidInvertedPendulumSystem(BaseSystem):
    """An inverted pendulum on a cart that a PID controller of its angle keeps upright.

    The controller sets the force on the cart to `kp e + ki ∫ e dt - kd theta'` from the error `e = -theta` of the
    angle, within the bounds of the force. Near upright the closed loop is stable when `kp > (M + m) g` and `kd > 0`,
    and the angle returns to upright while the cart drifts. The system has the angle `cart.theta`, the position
    `cart.x`, the force `cart.force`, and the observable `tilt`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.control import PidInvertedPendulumSystem
        >>> PidInvertedPendulumSystem(kp=50.0).prepare().system_.interactions[0].kp
        50.0
    """

    field = 'engineering/control'
    topics = ('control', 'feedback', 'pid', 'unstable equilibrium', 'ode')
    references = ('Åström and Murray 2008',)
    until = 5.0
    dt = 0.05
    solver = RK4Solver(dt=0.001)
    observables = (ObservableSpec('tilt', _tilt),)

    def __init__(
        self,
        theta: float = 0.1,
        *,
        kp: float = 40.0,
        ki: float = 1.0,
        kd: float = 8.0,
        M: float = 1.0,
        m: float = 0.1,
        l: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.theta = theta
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.M = M
        self.m = m
        self.l = l

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        cart = CartPendulumElement('cart', theta=self.theta, M=self.M, m=self.m, l=self.l)
        controller = PIDInteraction(
            'pid',
            plant='cart',
            integrator='memory',
            control='force',
            measure='theta',
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            limits=(-FORCE_LIMIT, FORCE_LIMIT),
        )
        return System(self.label, elements=[cart, IntegratorElement('memory')], interactions=[controller])


class LqrOscillatorSystem(BaseSystem):
    """A harmonic oscillator that a linear-quadratic regulator brings to rest.

    The gain `K` of the regulator minimises `∫ (q (x^2 + x'^2) + r force^2) dt` for the linearisation of the oscillator,
    which `linearization` computes from the Riccati equation, and the state feedback sets `force = -K (x, x')`. A
    smaller weight `r` of the force gives a larger gain and a faster return to rest. The system has the displacement
    `oscillator.x`, the force `oscillator.force`, and the observable `energy`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.control import LqrOscillatorSystem
        >>> [round(value, 3) for value in LqrOscillatorSystem().prepare().system_.interactions[0].gain]
        [2.317, 3.825]
    """

    field = 'engineering/control'
    topics = ('control', 'feedback', 'linear-quadratic regulator', 'oscillator', 'ode')
    references = ('Åström and Murray 2008',)
    until = 10.0
    dt = 0.05
    solver = RK4Solver(dt=0.001)
    observables = (ObservableSpec('energy', _oscillator_energy),)

    def __init__(
        self,
        x: float = 1.0,
        *,
        omega: float = 1.0,
        zeta: float = 0.0,
        q: float = 1.0,
        r: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.omega = omega
        self.zeta = zeta
        self.q = q
        self.r = r

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        oscillator = DrivenOscillatorElement('oscillator', x=self.x, omega=self.omega, zeta=self.zeta)
        linear = linearization(
            System(elements=[oscillator]),
            controls=['oscillator.force'],
            state_weights=self.q * np.eye(2),
            control_weights=[[self.r]],
        )
        gain = linear.gain
        if gain is None:
            error_msg = 'The linearization of the oscillator gives no gain, since it has no cost weights.'
            raise ValidationError(error_msg)
        states = tuple(name.removeprefix('oscillator.') for name in linear.states)
        regulator = StateFeedbackInteraction(
            'regulator',
            plant='oscillator',
            control='force',
            states=states,
            gain=tuple(float(value) for value in gain[0]),
            limits=(-FORCE_LIMIT, FORCE_LIMIT),
        )
        return System(self.label, elements=[oscillator], interactions=[regulator])
