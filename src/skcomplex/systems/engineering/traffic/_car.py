"""Car on a road whose position `x` advances at its speed `v`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import StateSpec, VisualSpec


class CarElement(Element):
    """Car on a road whose position `x` advances at its speed `v`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(role='position')] = 0.0,
        v: Annotated[object, StateSpec(role='velocity')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the advance of the position at the speed."""
        return [RateLaw(s.x, s.v)]
