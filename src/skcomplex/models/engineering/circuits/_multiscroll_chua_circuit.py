"""Chua circuit with a sine nonlinearity that gives attractors of many scrolls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import where
from ....specs import ParameterSpec, StateSpec, VisualSpec


class MultiscrollChuaCircuitElement(Element):
    """Chua circuit with a sine nonlinearity that gives attractors of many scrolls."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 1.0,
        z: Annotated[object, StateSpec()] = 0.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.82,
        beta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 14.286,
        a: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
        b: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.11,
        c: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.c = c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the states of the circuit."""
        return [
            RateLaw(s.x, s.alpha * (s.y - MultiscrollChuaCircuitElement._scroll_nonlinearity(s.x, s.a, s.b, s.c))),
            RateLaw(s.y, s.x - s.y + s.z),
            RateLaw(s.z, -s.beta * s.y),
        ]

    @staticmethod
    def _scroll_nonlinearity(x: Expression, a: Expression, b: Expression, c: Expression) -> Expression:
        """Return the sine characteristic of the multiscroll Chua circuit, continued linearly beyond `|x| = 2 a c`."""
        edge = 2 * a * c
        slope = b * pi / (2 * a)
        return where(x >= edge, slope * (x - edge), where(x <= -edge, slope * (x + edge), -b * sin(pi * x / (2 * a))))
