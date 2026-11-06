"""Node of a network that carries a load of shortest paths up to its capacity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class LoadedNodeElement(Element):
    """Node of a network that carries a load, the shortest paths through it, and fails above its capacity.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        load:
            The betweenness of the node on the network present, read at the start of every step.

        capacity:
            The largest load the node carries, a negative value for a node that fails at the first step.

        giant:
            The number of nodes of the largest connected component, the same for every node.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import LoadedNodeElement
        >>> LoadedNodeElement('nodes', capacity=12.0).capacity
        12.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        load: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        capacity: Annotated[float, StateSpec()] = 0.0,
        giant: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.load = load
        self.capacity = capacity
        self.giant = giant
