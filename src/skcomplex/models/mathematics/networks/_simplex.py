"""Triangle of a growing simplicial complex, with its age in steps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class SimplexElement(Element):
    """Triangle of a growing simplicial complex, with its age in steps.

    Args:
        label:
            The label of the triangle or the group on the triangles.

        visual:
            How the triangle is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the triangle joined.

    Attributes:
        default_visual_:
            The visual of the triangle when it sets none.

    Examples:
        >>> from skcomplex.models.mathematics.networks import SimplexElement
        >>> SimplexElement('triangles').age
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age
