"""Node of a growing network, with its age in steps and its degree."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})


class VertexElement(Element):
    """Node of a growing network, with its age in steps and its degree.

    New nodes start with age 0, and the maps of the model age the node and read its degree from the edges at the start
    of every step.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

    Examples:
        >>> from skcomplex.models.mathematics.networks import VertexElement
        >>> VertexElement('nodes').age
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = DEGREE_VISUAL,
        *,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age
        self.degree = degree
