"""Node of a network that is susceptible at `i = 0` or infected at `i = 1`, in continuous time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.specs import ParameterSpec, StateSpec, VisualSpec


class HostElement(Element):
    """Node of a network that is susceptible at `i = 0` or infected at `i = 1`, in continuous time.

    A `SpreadingInteraction` infects and recovers it, and a `TopologyInteraction` rewires its edges at its rate `w`.

    Args:
        label:
            The label of the node or of the group of nodes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        i:
            Whether the node is infected.

        w:
            The rate at which the node, while susceptible, rewires an edge to an infected neighbour.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import HostElement
        >>> HostElement('node').w
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        i: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        w: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.i = i
        self.w = w
