"""Oscillator of a tunnel diode with a cubic current, of the van der Pol type."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class TunnelDiodeCircuitElement(Element):
    """Oscillator of a tunnel diode with a cubic current, of the van der Pol type."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        V: Annotated[object, StateSpec()] = 0.1,
        i: Annotated[object, StateSpec()] = 0.0,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        gamma: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.V = V
        self.i = i
        self.L = L
        self.C = C
        self.alpha = alpha
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the states of the circuit."""
        return [
            RateLaw(s.V, -(s.gamma * s.V**3 - s.alpha * s.V + s.i) / s.C),
            RateLaw(s.i, s.V / s.L),
        ]
