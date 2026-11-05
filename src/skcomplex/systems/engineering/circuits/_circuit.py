"""RLC circuit whose charge follows `L q'' + R q' + q / C = V cos(ω t)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, EquationLaw
from ....core.components.laws.utils import derivative
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CircuitElement(Element):
    """RLC circuit whose charge follows `L q'' + R q' + q / C = V cos(ω t)`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        q: Annotated[object, StateSpec(order=2)] = 1.0,
        q_d1: float = 0.0,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        R: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        V: Annotated[object, ParameterSpec()] = 0.0,
        omega: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.q = q
        self.q_d1 = q_d1
        self.L = L
        self.R = R
        self.C = C
        self.V = V
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the equation of the charge."""
        return [
            EquationLaw(s.L * derivative(s.q, 2) + s.R * derivative(s.q, 1) + s.q / s.C == s.V * cos(s.omega * s.t)),
        ]
