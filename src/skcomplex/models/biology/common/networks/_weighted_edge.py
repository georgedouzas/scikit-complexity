"""Define weighted edges, members of a group on the edges of a network that carry a weight drawn when they appear."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from .....core.elements import Element
from .....specs import StateSpec, VisualSpec


class WeightedEdgeElement(Element):
    """Edge of a network with a weight `w`, such as the strength of an influence or a coupling between two species.

    A new edge is marked `fresh` until an `EdgeDrawInteraction` draws its weight.

    Args:
        label:
            The label of the edge or of the group of edges.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        w:
            The weight.

        fresh:
            Whether the edge has not drawn its weight yet, 1 or 0.

    Examples:
        >>> from skcomplex.models.biology.common.networks import WeightedEdgeElement
        >>> WeightedEdgeElement('edge').fresh
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        w: Annotated[float, StateSpec()] = 0.0,
        fresh: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.w = w
        self.fresh = fresh
