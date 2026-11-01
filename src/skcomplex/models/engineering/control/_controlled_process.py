"""First-order process under a PID controller whose derivative term is solved into the time constant."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import step
from ....specs import ParameterSpec, StateSpec, VisualSpec

DISTURBANCE_TIME = 10.0


CHANGE_TIME = 20.0


class ControlledProcessElement(Element):
    """First-order process under a PID controller whose derivative term is solved into the time constant."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[object, StateSpec()] = 0.0,
        integral: Annotated[object, StateSpec()] = 0.0,
        tau: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        gain: Annotated[object, ParameterSpec()] = 1.0,
        Kp: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        Ki: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        Kd: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        setpoint: Annotated[object, ParameterSpec()] = 1.0,
        change: Annotated[object, ParameterSpec()] = 0.0,
        disturbance: Annotated[object, ParameterSpec()] = -0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.integral = integral
        self.tau = tau
        self.gain = gain
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.change = change
        self.disturbance = disturbance

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the output `y` and of the integral of the error."""
        return [
            RateLaw(
                s.y,
                (
                    -s.y
                    + s.gain * (s.Kp * ControlledProcessElement._error(s) + s.Ki * s.integral)
                    + s.disturbance * step(s.t, DISTURBANCE_TIME)
                )
                / (s.tau + s.gain * s.Kd),
            ),
            RateLaw(s.integral, ControlledProcessElement._error(s)),
        ]

    @staticmethod
    def _error(s: SimpleNamespace) -> Expression:
        """Return the error of the controlled process, the set point after its change less the output."""
        return s.setpoint + s.change * step(s.t, CHANGE_TIME) - s.y
