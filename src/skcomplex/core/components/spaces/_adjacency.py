"""Define the neighbourhood of the nodes joined to a node by edges of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseNeighborhood


class AdjacencyNeighborhood(BaseNeighborhood):
    """A neighbourhood of the nodes joined to a node by the edges of a network.

    The neighbours come in node order, after the node itself when `center` includes it. A local rule receives the values
    of the neighbours padded to the largest degree of the graph, the largest number of edges at a node. On a directed
    graph the neighbours are the successors.

    Unlike `BallNeighborhood`, which reads offsets on a lattice, it reads the edges of a graph, so each node may have a
    different number of neighbours. Choose it when members sit on the nodes of a network.

    It appears in many fields. A person catches a disease from their contacts, a voter copies the opinion of a friend,
    and a bus of a power grid exchanges power with the buses it is wired to.

    Args:
        center:
            Whether the node itself comes first.

        layer:
            The label of the layer whose edges join the neighbours, or `None` for the graph of the network.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Examples:
        >>> from skcomplex.core.components.spaces import AdjacencyNeighborhood
        >>> AdjacencyNeighborhood(layer='work').layer
        'work'
    """

    def __init__(self, *, center: bool = False, layer: str | None = None, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.center = center
        self.layer = layer
