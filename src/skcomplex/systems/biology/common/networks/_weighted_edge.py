"""Define weighted edges of a network.

They are members of a group on the edges of a network that carry a weight drawn when they appear.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace
from .....core.types import Array


class WeightedEdgeElement(Element):
    """Edge of a network with a weight `w`.

    The weight is such as the strength of an influence or a coupling between two species.

    A new edge is marked `fresh` until an `EdgeDrawInteraction` draws its weight.

    Args:
        label:
            The label of the edge or of the group of edges.

        w:
            The weight.

        fresh:
            Whether the edge has not drawn its weight yet, 1 or 0.

        space:
            The network whose edges hold one member each, or `None` for one edge.

        on:
            `'edges'` for one member per edge of the network, or `None` for one per node.

    Examples:
        >>> from skcomplex.systems.biology.common.networks import WeightedEdgeElement
        >>> WeightedEdgeElement('edge').fresh
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        w: float | Array = 0.0,
        fresh: Annotated[float | Array, FiniteEnumeratedSet([0, 1])] = 1.0,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label, space=space, on=on)
        self.w = w
        self.fresh = fresh
