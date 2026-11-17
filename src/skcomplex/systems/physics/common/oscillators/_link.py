"""Edge of a network with a weight `w`.

Couplings along the edge read the weight.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.specs import ParameterSpec


class LinkElement(Element):
    """Edge of a network with a weight `w`.

    Couplings along the edge read the weight.

    Args:
        label:
            The label of the edge or the group on the edges.

        w:
            The weight of the edge.

    Attributes:
        default_visual_:
            How the edge is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.systems.physics.common.oscillators import LinkElement
        >>> LinkElement('link', w=0.5).w
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        w: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.w = w
