"""Room heated by a furnace towards a setting and leaking heat to an outside temperature with a daily swing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import cos, oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import clip
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

HOURS_PER_DAY = 24.0


class RoomElement(Element):
    """Room heated by a furnace towards a setting and leaking heat to an outside temperature with a daily swing."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[object, StateSpec()] = 10.0,
        setting: Annotated[object, ParameterSpec()] = 18.0,
        furnace: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        leak: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        outside: Annotated[object, ParameterSpec()] = 10.0,
        swing: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.setting = setting
        self.furnace = furnace
        self.leak = leak
        self.outside = outside
        self.swing = swing

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the saturated heating and the leak of the temperature `T`."""
        return [
            RateLaw(
                s.T,
                clip(s.setting - s.T, 0, s.furnace)
                - s.leak * (s.T - s.outside - s.swing * cos(2 * pi * (s.t - HOURS_PER_DAY / 2) / HOURS_PER_DAY)),
            ),
        ]
