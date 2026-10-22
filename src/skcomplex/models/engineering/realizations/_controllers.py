"""Define feedback controllers that set the control inputs of a plant from its states during a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.base import Expression, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Boundary, Law, Rate, Symbols, derivative, where


def _limited(value: Expression, limits: tuple[float, float] | None) -> Expression:
    """Return a value clipped to limits, or the value itself without limits."""
    if limits is None:
        return value
    lower, upper = limits
    return where(value > upper, upper, where(value < lower, lower, value))


def _read(plant: Symbols, name: str) -> Expression:
    """Return the symbol of a variable of a plant by its name, such as `x` or its velocity `x_d1`."""
    value: Expression = getattr(plant, name)
    return value


class Integrator(Element):
    """Element that holds the integral of a signal, whose rate a controller sets.

    Args:
        label:
            The label of the integrator.

        value:
            The initial value of the integral.

    Examples:
        >>> from skcomplex.models.engineering.realizations import Integrator
        >>> Integrator('memory').value
        0.0
    """

    value: float = state(default=0.0)


class PID(Interaction):
    """Proportional-integral-derivative controller that sets a control input of a plant from the error of a measure.

    The error is `e = setpoint - y` for the measured state `y`, and the controller sets the control input to
    `u = kp e + ki I - kd y'`, clipped to the limits, with the derivative taken on the measure. The integral
    `I' = e` of the error lives in an `Integrator` bound to the role `integrator`. The control input is an adaptive
    parameter of the plant, which the controller sets after every solver step, so it holds within a step like a
    sampled controller. The derivative term reads the first derivative of the measure, so a measure with a derivative
    term is a state of order 2.

    Args:
        label:
            The label of the controller.

        plant:
            The path of the element that holds the measure and the control input.

        integrator:
            The path of the integrator of the error.

        control:
            The name of the adaptive parameter of the plant that the controller sets.

        measure:
            The name of the measured state of the plant.

        kp:
            The proportional gain.

        ki:
            The integral gain.

        kd:
            The derivative gain.

        setpoint:
            The value the measure is kept at.

        limits:
            The lower and upper limit of the control input, or `None` for no limits.

    Examples:
        >>> from skcomplex.models.engineering.realizations import PID
        >>> PID('pid', plant='cart', integrator='memory', control='force', measure='theta', kp=40.0).kp
        40.0
    """

    plant: str = role()
    integrator: str = role()
    control: str = 'u'
    measure: str = 'y'
    kp: float = parameter(default=1.0)
    ki: float = parameter(default=0.0)
    kd: float = parameter(default=0.0)
    setpoint: float = parameter(default=0.0)
    limits: tuple[float, float] | None = None

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate of the integral of the error and the control input it sets.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        measured = _read(s.plant, self.measure)
        error = s.setpoint - measured
        output = s.kp * error + s.ki * s.integrator.value
        if self.kd != 0:
            output -= s.kd * derivative(measured, 1)
        return [Rate(s.integrator.value, error), Boundary(_read(s.plant, self.control), _limited(output, self.limits))]


class StateFeedback(Interaction):
    """Linear state feedback `u = -K x` that sets a control input of a plant from its states.

    The control input is an adaptive parameter of the plant, which the controller sets after every solver step, so it
    holds within a step like a sampled controller. A gain from `lqr` of a linearisation of the plant makes the
    controller a linear-quadratic regulator.

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
        >>> from skcomplex.models.engineering.realizations import StateFeedback
        >>> StateFeedback('regulator', plant='mass', control='force', states=('x', 'x_d1'), gain=(1.0, 1.7)).gain
        (1.0, 1.7)
    """

    plant: str = role()
    control: str = 'u'
    states: Sequence[str] = ()
    gain: Sequence[float] = ()
    limits: tuple[float, float] | None = None

    def laws(self, s: Symbols) -> list[Law]:
        """Return the control input the feedback sets.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        feedback = -sum(
            (float(gain) * _read(s.plant, name) for name, gain in zip(self.states, self.gain, strict=True)),
            0,
        )
        return [Boundary(_read(s.plant, self.control), _limited(feedback, self.limits))]
