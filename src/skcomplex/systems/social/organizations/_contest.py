"""Contest element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ContestElement(Element):
    """Contest element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='resources')] = 55.0,
        b: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='resources')] = 45.0,
        budget: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        lifetime: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b
        self.budget = budget
        self.lifetime = lifetime
