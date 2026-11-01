"""Robot in a Gaussian light field that slows down in light, reacting to the light it measured a delay ago."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos, exp, oo, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw
from ....core.laws.utils import delayed, where
from ....specs import ParameterSpec, StateSpec, VisualSpec


class LightSensitiveRobotElement(Element):
    """Robot in a Gaussian light field that slows down in light, reacting to the light it measured a delay ago.

    The robot moves at the speed `v = V_inf + (V0 - V_inf) exp(-I(t - delay))` in the direction `phi`, which diffuses
    with time scale `tau`, `dphi = sqrt(2 / tau) dW`. The light is `I(x, y) = I0 exp(-(x^2 + y^2) / r0^2)`, and a wall
    at radius `R` reflects the robot back into a circular well.

    Args:
        label:
            The label of the robot or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial horizontal position.

        y:
            The initial vertical position.

        phi:
            The initial direction of motion.

        V0:
            The speed in the dark.

        V_inf:
            The speed in strong light.

        I0:
            The intensity at the centre of the light.

        r0:
            The width of the light.

        tau:
            The time scale of rotational diffusion.

        R:
            The radius of the well.

        delay:
            The sensory delay, 0 for a robot that reacts to the light it measures now.

    Examples:
        >>> from skcomplex.models.engineering.robotics import LightSensitiveRobotElement
        >>> LightSensitiveRobotElement('robot', delay=5.0).delay
        5.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        phi: Annotated[float, StateSpec()] = 0.0,
        V0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        V_inf: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        I0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        R: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        delay: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.phi = phi
        self.V0 = V0
        self.V_inf = V_inf
        self.I0 = I0
        self.r0 = r0
        self.tau = tau
        self.R = R
        self.delay = delay

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the motion at the light-dependent speed, the rotational noise, and the reflecting wall.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        sensed = (
            LightSensitiveRobotElement._light(s, delayed(s.x, self.delay), delayed(s.y, self.delay))
            if self.delay
            else LightSensitiveRobotElement._light(s, s.x, s.y)
        )
        speed = s.V_inf + (s.V0 - s.V_inf) * exp(-sensed)
        reach = where(sqrt(s.x**2 + s.y**2) > s.R, sqrt(s.x**2 + s.y**2), s.R)
        return [
            RateLaw(s.x, speed * cos(s.phi)),
            RateLaw(s.y, speed * sin(s.phi)),
            NoiseLaw(s.phi, sqrt(2 / s.tau)),
            BoundaryLaw(s.x, s.x * s.R / reach),
            BoundaryLaw(s.y, s.y * s.R / reach),
        ]

    @staticmethod
    def _light(s: SimpleNamespace, x: Expression, y: Expression) -> Expression:
        """Return the Gaussian light intensity `I0 exp(-(x^2 + y^2) / r0^2)` at a point."""
        return s.I0 * exp(-(x**2 + y**2) / s.r0**2)
