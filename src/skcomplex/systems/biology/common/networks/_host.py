"""Node of a network that is susceptible or infected, in continuous time.

It is susceptible at `i = 0` and infected at `i = 1`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace
from .....core.types import Array


class HostElement(Element):
    """Node of a network that is susceptible or infected, in continuous time.

    It is susceptible at `i = 0` and infected at `i = 1`.

    A `SpreadingInteraction` infects and recovers it, and a rewiring interaction rewires its edges at its rate `w`.

    Args:
        label:
            The label of the node or of the group of nodes.

        i:
            Whether the node is infected.

        w:
            The rate at which the node, while susceptible, rewires an edge to an infected neighbour.

        space:
            The network whose nodes hold one member each, or `None` for one node.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import HostElement
        >>> HostElement('node').w
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        i: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.0,
        w: Annotated[float | Array, RealSet.closed_open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.i = i
        self.w = w
