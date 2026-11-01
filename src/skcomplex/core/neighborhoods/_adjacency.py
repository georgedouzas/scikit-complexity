"""Define adjacency neighbourhoods, the nodes along the edges of a node of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseNeighborhood


class AdjacencyNeighborhood(BaseNeighborhood):
    """Nodes along the edges of a node of a network, in node order, after the node itself when it is included.

    Rules receive the values of the neighbours padded to the largest degree of the graph. On a directed graph the
    neighbours are the successors.

    Args:
        center:
            Whether the node itself comes first.

        layer:
            The label of the layer whose edges join the neighbours, or `None` for the graph of the network.

        label:
            The label of the neighbourhood.

    Examples:
        >>> from skcomplex.core.neighborhoods import AdjacencyNeighborhood
        >>> AdjacencyNeighborhood(layer='work').layer
        'work'
    """

    def __init__(self, *, center: bool = False, layer: str | None = None, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.center = center
        self.layer = layer
