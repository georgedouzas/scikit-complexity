"""Node of a growing network, with its age in steps and its degree."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace
from .....core.specs import VisualSpec

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})


class VertexElement(Element):
    """Node of a growing network, with its age in steps and its degree.

    New nodes start with the age their growth gives, and the maps of the system age the node and read its degree from
    the edges at the start of every step.

    Args:
        label:
            The label of the node group.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import VertexElement
        >>> VertexElement('nodes').age
        0.0
    """

    visual = DEGREE_VISUAL

    def __init__(
        self,
        label: str,
        *,
        age: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        degree: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.age = age
        self.degree = degree
