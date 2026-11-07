"""Displacement and parameters of the forced Toda oscillator, whose exponential spring is stiff on one side only."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import derivative
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class TodaElement(Element):
    """Displacement and parameters of the forced Toda oscillator, whose exponential spring is stiff on one side only.

    The acceleration is `x'' = -d x' - (e^x - 1) + a sin(omega t)`. The restoring force `e^x - 1` grows exponentially
    for `x > 0` and saturates at `-1` for `x < 0`, so the oscillator is passive and asymmetric, and its resonance curve
    bends towards low frequencies.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        x_d1:
            The initial velocity.

        d:
            The damping.

        a:
            The amplitude of the forcing.

        omega:
            The frequency of the forcing.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import TodaElement
        >>> TodaElement('toda').d
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        a: Annotated[float, ParameterSpec()] = 0.88,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.d = d
        self.a = a
        self.omega = omega

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the acceleration of the oscillator.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate of the second derivative of `x`.
        """
        s = namespace
        return [RateLaw(s.x, -s.d * derivative(s.x, 1) - (exp(s.x) - 1) + s.a * sin(s.omega * s.t))]
