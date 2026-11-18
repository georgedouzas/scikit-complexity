"""Edge of a network with a weight `w`.

Couplings along the edge read the weight.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace


class LinkElement(Element):
    """Edge of a network with a weight `w`.

    Couplings along the edge read the weight, as the strength of a power line or of a synapse.

    Args:
        label:
            The label of the edge or the group on the edges.

        w:
            The weight of the edge, or one weight per edge.

        space:
            The network of the edges, or `None` for one edge.

        on:
            `'edges'` for one member on every edge of the network, `'hyperedges'` for one on every hyperedge, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.common.oscillators import LinkElement
        >>> LinkElement('link', w=0.5).w
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        w: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label, space=space, on=on)
        self.w = w
