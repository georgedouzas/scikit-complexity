"""Stochastic fold `dx = (a x^3 + b x + phi(t)) dt + sigma dW` whose control rises linearly from `phi0` to `phi1`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.laws.utils import ramp
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RampedFoldElement(Element):
    """Stochastic fold `dx = (a x^3 + b x + phi(t)) dt + sigma dW` whose control rises linearly from `phi0` to `phi1`.

    For `a < 0 < b` the fold has two stable branches for `|phi| < phi_c = sqrt(-4 b^3 / (27 a))`, and the lower
    branch vanishes at `phi = phi_c`. The control rises at the rate `rate` and stays at `phi1` once it reaches it.

    Args:
        label:
            The label of the system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial state.

        a:
            The cubic coefficient, negative.

        b:
            The linear coefficient, positive.

        sigma:
            The amplitude of the noise.

        phi0:
            The initial control.

        phi1:
            The final control.

        rate:
            The rate at which the control rises.

    Examples:
        >>> from skcomplex.models.earth.climate import RampedFoldElement
        >>> RampedFoldElement('system').rate
        0.004
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = -1.3247,
        a: Annotated[float, ParameterSpec()] = -1.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        phi0: float = -1.0,
        phi1: float = 1.0,
        rate: float = 0.004,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a
        self.b = b
        self.sigma = sigma
        self.phi0 = phi0
        self.phi1 = phi1
        self.rate = rate

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the fold with its ramped control and its noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        control = self.phi0 + ramp(s.t, 0.0, (self.phi1 - self.phi0) / self.rate, self.phi1 - self.phi0)
        return [RateLaw(s.x, s.a * s.x**3 + s.b * s.x + control), NoiseLaw(s.x, s.sigma)]
