"""Define feedback controllers that set the control inputs of a plant from its states during a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, BoundaryLaw, RateLaw
from ....core.laws.utils import derivative, where
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


def _limited(value: Expression, limits: tuple[float, float] | None) -> Expression:
    """Return a value clipped to limits, or the value itself without limits."""
    if limits is None:
        return value
    lower, upper = limits
    return where(value > upper, upper, where(value < lower, lower, value))


def _read(plant: SimpleNamespace, name: str) -> Expression:
    """Return the symbol of a variable of a plant by its name, such as `x` or its velocity `x_d1`."""
    value: Expression = getattr(plant, name)
    return value


class IntegratorElement(Element):
    """Element that holds the integral of a signal, whose rate a controller sets.

    Args:
        label:
            The label of the integrator.

        value:
            The initial value of the integral.

    Examples:
        >>> from skcomplex.models.engineering.realizations import IntegratorElement
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
    """Proportional-integral-derivative controller that sets a control input of a plant from the error of a measure.

    The error is `e = setpoint - y` for the measured state `y`, and the controller sets the control input to
    `u = kp e + ki I - kd y'`, clipped to the limits, with the derivative taken on the measure. The integral
    `I' = e` of the error lives in an `IntegratorElement` bound to the role `integrator`. The control input is an
    adaptive parameter of the plant, which the controller sets after every solver step, so it holds within a step like a
    sampled controller. The derivative term reads the first derivative of the measure, so a measure with a derivative
    term is a state of order 2.

    Args:
        label:
            The label of the controller.

        control:
            The name of the adaptive parameter of the plant that the controller sets.

        measure:
            The name of the measured state of the plant.

        limits:
            The lower and upper limit of the control input, or `None` for no limits.

        **values:
            The values of the declared variables, by name:

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

    Examples:
        >>> from skcomplex.models.engineering.realizations import PIDInteraction
        >>> PIDInteraction('pid', plant='cart', integrator='memory', control='force', measure='theta', kp=40.0).kp
        40.0
    """

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
        measured = _read(s.plant, self.measure)
        error = s.setpoint - measured
        output = s.kp * error + s.ki * s.integrator.value
        if self.kd != 0:
            output -= s.kd * derivative(measured, 1)
        return [
            RateLaw(s.integrator.value, error),
            BoundaryLaw(_read(s.plant, self.control), _limited(output, self.limits)),
        ]


class StateFeedbackInteraction(Interaction):
    """Linear state feedback `u = -K x` that sets a control input of a plant from its states.

    The control input is an adaptive parameter of the plant, which the controller sets after every solver step, so it
    holds within a step like a sampled controller. A gain from `lqr` of a linearisation of the plant makes the
    controller a linear-quadratic regulator.

    Args:
        label:
            The label of the controller.

        control:
            The name of the adaptive parameter of the plant that the controller sets.

        states:
            The names of the fed back states of the plant, such as `('x', 'x_d1')` for a position and its velocity.

        gain:
            The gain of every state, in the order of `states`.

        limits:
            The lower and upper limit of the control input, or `None` for no limits.

        **values:
            The values of the declared variables, by name:

            plant:
                The path of the element that holds the states and the control input.

    Examples:
        >>> from skcomplex.models.engineering.realizations import StateFeedbackInteraction
        >>> StateFeedbackInteraction(
        ...     'regulator', plant='mass', control='force', states=('x', 'x_d1'), gain=(1.0, 1.7)
        ... ).gain
        (1.0, 1.7)
    """

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
            (float(gain) * _read(s.plant, name) for name, gain in zip(self.states, self.gain, strict=True)),
            0,
        )
        return [BoundaryLaw(_read(s.plant, self.control), _limited(feedback, self.limits))]
