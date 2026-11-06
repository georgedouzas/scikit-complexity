"""Lead car whose speed swings about a cruising speed `u` with an amplitude and a frequency."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LeadCarElement(Element):
    """Lead car whose speed swings about a cruising speed `u` with an amplitude and a frequency."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(role='position')] = 0.0,
        v: Annotated[object, StateSpec(role='velocity')] = 1.0,
        u: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        amplitude: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        omega: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.u = u
        self.amplitude = amplitude
        self.omega = omega

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the advance of the position and the forced swing of the speed."""
        return [RateLaw(s.x, s.v), RateLaw(s.v, -s.u * s.amplitude * s.omega * cos(s.omega * s.t))]
