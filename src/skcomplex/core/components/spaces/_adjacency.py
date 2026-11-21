"""Define the neighbourhood of the nodes joined to a node by edges of a network."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseNeighborhood


class AdjacencyNeighborhood(BaseNeighborhood):
    """A neighbourhood of the nodes joined to a node by the edges of a network.

    A law that reads it sees the values of every node joined to a member by an edge, such as the friends of a voter in
    `AdjacencyNeighborhood(layer='friends')`. The neighbours come in node order, after the node itself when `center` is
    true, and on a directed graph they are the successors. A local rule receives them padded to the largest number of
    edges at a node. A person catches a disease from their contacts, a voter copies a friend, and a bus of a power grid
    trades power with the buses wired to it.

    Read more in the user guide, under Spaces.

    Args:
        center:
            Whether the node itself comes first.

        layer:
            The label of the layer whose edges join the neighbours, or `None` for the graph of the network.

        label:
            The name of the neighbourhood, such as `'moore'`.

    See Also:
        NetworkSpace: Holds the graph and the layers whose edges it reads.
        RadiusNeighborhood: Reads the members within a distance on a lattice or in a box.

    Examples:
        >>> from skcomplex.core.components.spaces import AdjacencyNeighborhood
        >>> AdjacencyNeighborhood(layer='work').layer
        'work'
    """

    def __init__(self, *, center: bool = False, layer: str | None = None, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.center = center
        self.layer = layer

    def _count_centers(self, n_dim: int | None = None) -> int:
        """Return one when the node itself comes first, and zero otherwise."""
        return int(self.center)
