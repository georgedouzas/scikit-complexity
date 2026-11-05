"""Stage of a process, a stock of material `m` with a constant feed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class StageElement(Element):
    """Stage of a process, a stock of material `m` with a constant feed."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        m: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        feed: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.m = m
        self.feed = feed

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the constant feed of the stage."""
        return [RateLaw(s.m, s.feed)]
