"""Performance element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class PerformanceElement(Element):
    """Performance element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        performance: Annotated[object, StateSpec(quantity='performance')] = 1.0,
        perceived: Annotated[object, StateSpec(quantity='performance')] = 1.0,
        desired: Annotated[object, StateSpec(quantity='performance')] = 1.0,
        bias: Annotated[object, ParameterSpec(domain=RealSet.open(-oo, 1))] = 0.1,
        perception_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        correction_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        goal_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        erosion: Annotated[object, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.performance = performance
        self.perceived = perceived
        self.desired = desired
        self.bias = bias
        self.perception_time = perception_time
        self.correction_time = correction_time
        self.goal_time = goal_time
        self.erosion = erosion
