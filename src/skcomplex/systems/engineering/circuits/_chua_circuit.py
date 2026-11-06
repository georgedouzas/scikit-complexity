"""Chua circuit with the piecewise-linear current of its diode."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ChuaCircuitElement(Element):
    """Chua circuit with the piecewise-linear current of its diode."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        z: Annotated[object, StateSpec()] = 0.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 15.6,
        beta: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 28.0,
        m0: Annotated[object, ParameterSpec()] = -8 / 7,
        m1: Annotated[object, ParameterSpec()] = -5 / 7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.alpha = alpha
        self.beta = beta
        self.m0 = m0
        self.m1 = m1

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the states of the circuit."""
        return [
            RateLaw(s.x, s.alpha * (s.y - s.x - s.m1 * s.x - (s.m0 - s.m1) * (abs(s.x + 1) - abs(s.x - 1)) / 2)),
            RateLaw(s.y, s.x - s.y + s.z),
            RateLaw(s.z, -s.beta * s.y),
        ]
