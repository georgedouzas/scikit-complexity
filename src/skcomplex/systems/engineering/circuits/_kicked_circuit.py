"""RLC circuit ringing down freely between the impulses of strength `A` that kick its rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, EquationLaw
from ....core.components.laws.utils import derivative
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class KickedCircuitElement(Element):
    """RLC circuit ringing down freely between the impulses of strength `A` that kick its rate."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.0,
        L: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        R: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        A: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.L = L
        self.R = R
        self.C = C
        self.A = A

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the free equation of the circuit between impulses."""
        return [EquationLaw(s.L * s.C * derivative(s.x, 2) + s.R * s.C * derivative(s.x, 1) + s.x == 0)]
