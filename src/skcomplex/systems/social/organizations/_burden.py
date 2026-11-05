"""Burden element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class BurdenElement(Element):
    """Burden element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        capability: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='capability')] = 0.5,
        intervention: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='intervention')] = 0.0,
        desired: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        fix_rate: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        build_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        solution_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        erosion: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.capability = capability
        self.intervention = intervention
        self.desired = desired
        self.fix_rate = fix_rate
        self.build_time = build_time
        self.solution_delay = solution_delay
        self.erosion = erosion
