"""Arms race element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ArmsRaceElement(Element):
    """Arms race element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='arms')] = 10.0,
        b: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='arms')] = 10.5,
        margin: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        adjustment_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        perception_delay: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b
        self.margin = margin
        self.adjustment_time = adjustment_time
        self.perception_delay = perception_delay
