"""Oscillator built from a memristor, whose current depends on the flux through it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class MemristorCircuitElement(Element):
    """Oscillator built from a memristor, whose current depends on the flux through it."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[object, StateSpec()] = 0.1,
        i: Annotated[object, StateSpec()] = 0.0,
        w: Annotated[object, StateSpec()] = 0.1,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        beta: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.i = i
        self.w = w
        self.C = C
        self.L = L
        self.alpha = alpha
        self.beta = beta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the states of the circuit."""
        return [
            RateLaw(s.v, s.i / s.C),
            RateLaw(s.i, -(s.v + s.beta * (s.w**2 - 1) * s.i) / s.L),
            RateLaw(s.w, -s.i - s.alpha * s.w + s.w * s.i),
        ]
