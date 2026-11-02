"""Pendulum swung up by a bang-bang control of its energy, with `f` integrating the control."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, oo, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative, sign
from ....specs import ParameterSpec, StateSpec, VisualSpec


class SwingingPendulumElement(Element):
    """Pendulum swung up by a bang-bang control of its energy, with `f` integrating the control."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec(order=2)] = 0.0,
        theta_d1: float = 0.0,
        f: Annotated[object, StateSpec(order=2)] = 0.0,
        f_d1: float = 0.0,
        omega: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5652475842498528,
        mu: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.f = f
        self.f_d1 = f_d1
        self.omega = omega
        self.mu = mu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the controlled rate of the angle and the rate of the control integral."""
        return [
            RateLaw(
                s.theta,
                -(s.omega**2) * sin(s.theta) - SwingingPendulumElement._swing_up_control(s) * cos(s.theta),
            ),
            RateLaw(s.f, SwingingPendulumElement._swing_up_control(s)),
        ]

    @staticmethod
    def _pendulum_energy(s: SimpleNamespace) -> Expression:
        """Return the energy per unit inertia of the pendulum, `θ'^2 / 2 - ω^2 cos θ`."""
        return derivative(s.theta, 1) ** 2 / 2 - s.omega**2 * cos(s.theta)

    @staticmethod
    def _swing_up_control(s: SimpleNamespace) -> Expression:
        """Return the bang-bang control `u = -μ sign(θ' cos θ) sign(E_s - E)` of the energy of the pendulum."""
        gap = s.omega**2 - SwingingPendulumElement._pendulum_energy(s)
        return -s.mu * sign(derivative(s.theta, 1) * cos(s.theta)) * sign(gap)
