"""Page of a network with its PageRank, its degree, the number of pages, and the damping of the surfer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class NodeRankElement(Element):
    """Page of a network with its PageRank, its degree, the number of pages, and the damping of the surfer.

    Args:
        label:
            The label of the page or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        rank:
            The initial rank.

        damping:
            The probability that the surfer follows a link rather than jumping to a random page.

        degree:
            The number of links of the page.

        n_nodes:
            The number of pages of the network.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NodeRankElement
        >>> NodeRankElement('page', damping=0.9).damping
        0.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rank: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        damping: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.85,
        degree: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        n_nodes: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rank = rank
        self.damping = damping
        self.degree = degree
        self.n_nodes = n_nodes
