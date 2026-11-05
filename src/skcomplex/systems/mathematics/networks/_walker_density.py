"""Probability that a random walker is at a node of a hypergraph, with the weight `D` of the node."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class WalkerDensityElement(Element):
    """Probability that a random walker is at a node of a hypergraph, with the weight `D` of the node.

    Args:
        label:
            The label of the node or the group on the nodes.

        visual:
            How the node is drawn, or `None` for `default_visual_`.

        p:
            The probability of the walker at the node.

        D:
            The total weight of the moves out of the node, the sum of `(|e| - 1)^2` over its hyperedges `e`.

    Attributes:
        default_visual_:
            The visual of the node when it sets none.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import WalkerDensityElement
        >>> WalkerDensityElement('node', D=4.0).D
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        p: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.p = p
        self.D = D
