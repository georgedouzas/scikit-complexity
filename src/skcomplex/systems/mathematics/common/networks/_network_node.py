"""Nodes of a network.

The nodes carry their degree, clustering, betweenness, and community.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from .....core.components.elements import Element
from .....core.components.spaces import BaseSpace
from .....core.specs import VisualSpec


class NetworkNodeElement(Element):
    """Nodes of a network, each with its degree, clustering, betweenness, and community.

    Args:
        label:
            The label of the nodes.

        degree:
            The degree of every node, one per node or shared.

        clustering:
            The clustering coefficient of every node, one per node or shared.

        betweenness:
            The normalized betweenness centrality of every node, one per node or shared.

        community:
            The index of the community of every node, one per node or shared.

        size:
            The number of nodes, or `None` for one.

        space:
            The network with one node per site, or `None`.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import NetworkNodeElement
        >>> NetworkNodeElement('nodes', degree=[1.0, 2.0, 1.0], size=3).degree
        [1.0, 2.0, 1.0]
    """

    visual = VisualSpec('agents', {'color_by': 'community'})

    def __init__(
        self,
        label: str,
        *,
        degree: Annotated[object, NN] = 0.0,
        clustering: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        betweenness: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        community: Annotated[object, NN] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.degree = degree
        self.clustering = clustering
        self.betweenness = betweenness
        self.community = community
