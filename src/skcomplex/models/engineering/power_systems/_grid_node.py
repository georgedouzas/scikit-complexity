"""Node of a power grid whose phase follows the swing equation with a power and a damping."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative
from ....specs import ParameterSpec, StateSpec, VisualSpec


class GridNodeElement(Element):
    """Node of a power grid whose phase follows the swing equation with a power and a damping."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[object, StateSpec(order=2)] = 0.0,
        phi_d1: float = 0.0,
        P: Annotated[object, ParameterSpec()] = 1.0,
        gamma: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.phi_d1 = phi_d1
        self.P = P
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the swing equation of the phase."""
        return [RateLaw(s.phi, s.P - s.gamma * derivative(s.phi, 1))]
