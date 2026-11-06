"""Slow-fast Chua circuit whose oscillations mix small and large amplitudes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MixedModeChuaCircuitElement(Element):
    """Slow-fast Chua circuit whose oscillations mix small and large amplitudes."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.01,
        y: Annotated[object, StateSpec()] = 0.0,
        z: Annotated[object, StateSpec()] = 0.0,
        epsilon: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        a: Annotated[object, ParameterSpec()] = 1.5,
        b: Annotated[object, ParameterSpec()] = -1.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        R: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        beta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.005,
        c: Annotated[object, ParameterSpec()] = 0.00053,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.alpha = alpha
        self.R = R
        self.beta = beta
        self.c = c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the states of the circuit."""
        return [
            RateLaw(s.x, (-s.y + (s.a + s.b * s.x) * s.x**2) / s.epsilon),
            RateLaw(s.y, s.alpha * (s.x - s.R * s.y - s.z)),
            RateLaw(s.z, s.c - s.beta * s.y),
        ]
