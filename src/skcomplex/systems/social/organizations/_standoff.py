"""Standoff element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class StandoffElement(Element):
    """Standoff element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        stock: Annotated[object, StateSpec(quantity='stock')] = 50.0,
        effort_a: Annotated[object, StateSpec(quantity='effort')] = 0.0,
        effort_b: Annotated[object, StateSpec(quantity='effort')] = 0.0,
        effort_c: Annotated[object, StateSpec(quantity='effort')] = 0.0,
        goal_a: Annotated[object, ParameterSpec()] = 80.0,
        goal_b: Annotated[object, ParameterSpec()] = 20.0,
        goal_c: Annotated[object, ParameterSpec()] = 50.0,
        strength_a: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        strength_b: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        strength_c: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        intensification_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stock = stock
        self.effort_a = effort_a
        self.effort_b = effort_b
        self.effort_c = effort_c
        self.goal_a = goal_a
        self.goal_b = goal_b
        self.goal_c = goal_c
        self.strength_a = strength_a
        self.strength_b = strength_b
        self.strength_c = strength_c
        self.intensification_time = intensification_time
