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
from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BoundaryLaw, ConstraintLaw, RateLaw
from ....core.components.laws.utils import clip
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, Role, RoleSpec
from ....core.systems import BaseSystem, System

GRAVITY = 9.81
FORCE_LIMIT = 50.0


class CartPendulumInteraction(Interaction):
    """Motion of an inverted pendulum on a cart that a bounded horizontal force pushes.

    It follows Åström and Murray. The cart of mass `M` at position `x` carries a point mass `m` at the end of a rod of
    length `l`, at the angle `theta` from upright, positive towards negative `x`. The equations of motion are
    `(M + m) x'' - m l theta'' cos(theta) + m l theta'^2 sin(theta) = force` and
    `l theta'' - x'' cos(theta) = g sin(theta)`, so upright is unstable and a positive force tilts the pendulum back
    from a positive angle. The cart holds `x`, `theta`, the control input `force`, and its masses and length.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the cart.

        g:
            The acceleration of gravity.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.engineering.control import CartPendulumInteraction
        >>> cart = Element('cart', values={'x': 0.0, 'theta': 0.1, 'force': 0.0, 'M': 1.0, 'm': 0.1, 'l': 0.5})
        >>> System(elements=[cart], interactions=[CartPendulumInteraction('motion', target='cart')]).prepare().states_
        ('cart.x', "cart.x'", 'cart.theta', "cart.theta'")
    """

    law_classes = (ConstraintLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        g: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = GRAVITY,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.g = g

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the equations of motion of the cart and the pendulum."""
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        cart, t = context.target, context.t
        x_acceleration, theta_acceleration = cart.x.diff(t, 2), cart.theta.diff(t, 2)
        spin = cart.theta.diff(t)
        self.laws_: list[BaseComponent] = [
            ConstraintLaw(
                (cart.M + cart.m) * x_acceleration
                - cart.m * cart.l * theta_acceleration * cos(cart.theta)
                + cart.m * cart.l * spin**2 * sin(cart.theta)
                == cart.force,
            ),
            ConstraintLaw(
                cart.l * theta_acceleration - x_acceleration * cos(cart.theta) == context.g * sin(cart.theta),
            ),
        ]


class DrivenOscillatorInteraction(Interaction):
    """Motion of a damped harmonic oscillator that a force per unit mass drives.

    It follows `x'' = -omega^2 x - 2 zeta omega x' + force`, with the natural frequency `omega`, the damping ratio
    `zeta`, and the control input `force` of the element it binds. A controller that sets `force`, such as
    `StateFeedbackInteraction`, closes the loop, and without one `force` is a parameter that `linearization` takes as a
    control.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator, with the values `x`, `force`, `omega`, and `zeta`.

    Examples:
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.engineering.control import DrivenOscillatorInteraction
        >>> oscillator = Element('oscillator', values={'x': 1.0, 'force': 0.0, 'omega': 1.0, 'zeta': 0.0})
        >>> motion = DrivenOscillatorInteraction('motion', target='oscillator')
        >>> System(elements=[oscillator], interactions=[motion]).prepare().states_
        ('oscillator.x', "oscillator.x'")
    """

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the oscillator."""
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        oscillator, t = context.target, context.t
        x, omega = oscillator.x, oscillator.omega
        acceleration = -(omega**2) * x - 2 * oscillator.zeta * omega * x.diff(t) + oscillator.force
        self.laws_: list[BaseComponent] = [RateLaw(x.diff(t, 2), acceleration)]


class PIDInteraction(Interaction):
    """Proportional-integral-derivative controller of a control input of a plant.

    It sets the control input from the error of a measure. The error is `e = setpoint - y` for the measured state `y`,
    and the controller sets the control input to `u = kp e + ki I - kd y'`, clipped to the limits, with the derivative
    taken on the measure. The integral `I' = e` of the error is the value `value` of the element bound to the role
    `integrator`, such as `Element('memory', values={'value': 0.0})`. The controller sets the control input of the
    plant after every solver step, so it holds within a step like a sampled controller. The derivative term reads
    the first derivative of the measure, so a measure with a derivative term is a state of order 2.

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

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the integral of the error and the control input it sets."""
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s, value = context, context.integrator.value
        measured = getattr(s.plant, self.measure)
        error = s.setpoint - measured
        output = s.kp * error + s.ki * value
        if self.kd != 0:
            output -= s.kd * measured.diff(s.t)
        self.laws_: list[BaseComponent] = [
            RateLaw(value.diff(s.t), error),
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
            The names of the fed back states of the plant, with a prime for a derivative, such as `('x', "x'")` for a
            position and its velocity.

        gain:
            The gain of every state, in the order of `states`.

        limits:
            The lower and upper limit of the control input, or `None` for no limits.

    Examples:
        >>> from skcomplex.systems.engineering.control import StateFeedbackInteraction
        >>> StateFeedbackInteraction(
        ...     'regulator', plant='mass', control='force', states=('x', "x'"), gain=(1.0, 1.7)
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

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the control input the feedback sets."""
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        plant, t = context.plant, context.t
        read = [getattr(plant, name.rstrip("'")).diff(t, name.count("'")) for name in self.states]
        feedback = -sum((float(gain) * value for value, gain in zip(read, self.gain, strict=True)), 0)
        self.laws_: list[BaseComponent] = [
            BoundaryLaw(getattr(plant, self.control), clip(feedback, *(self.limits or (None, None)))),
        ]


def _get_tilt(s: SimpleNamespace) -> Expression:
    """Return the angle of the pendulum from upright."""
    tilt: Expression = s.cart.theta
    return tilt


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
    observables = (Observable('tilt', _get_tilt),)

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.theta = theta
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.M = M
        self.m = m
        self.l = l

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        values = {'x': 0.0, 'theta': self.theta, 'force': 0.0, 'M': self.M, 'm': self.m, 'l': self.l}
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
        self.system_ = System(
            self.label,
            elements=[Element('cart', values=values), Element('memory', values={'value': 0.0})],
            interactions=[controller, CartPendulumInteraction('motion', target='cart')],
            solver=RK4Solver(dt=0.001),
        )


def _get_oscillator_energy(s: SimpleNamespace) -> Expression:
    """Return the energy per unit mass of the oscillator, `(x'^2 + omega^2 x^2) / 2`."""
    oscillator = s.oscillator
    energy: Expression = (oscillator.x.diff(s.t) ** 2 + oscillator.omega**2 * oscillator.x**2) / 2
    return energy


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
    observables = (Observable('energy', _get_oscillator_energy),)

    def __init__(
        self,
        x: float = 1.0,
        *,
        omega: float = 1.0,
        zeta: float = 0.0,
        q: float = 1.0,
        r: float = 0.1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.omega = omega
        self.zeta = zeta
        self.q = q
        self.r = r

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        values = {'x': self.x, 'force': 0.0, 'omega': self.omega, 'zeta': self.zeta}
        oscillator = Element('oscillator', values=values)
        motion = DrivenOscillatorInteraction('motion', target='oscillator')
        linear = linearization(
            System(elements=[oscillator], interactions=[motion]),
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
        self.system_ = System(
            self.label,
            elements=[oscillator],
            interactions=[regulator, motion],
            solver=RK4Solver(dt=0.001),
        )
