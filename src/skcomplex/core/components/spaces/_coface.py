"""Define the neighbourhood of the faces around a link."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseNeighborhood


class CofaceNeighborhood(BaseNeighborhood):
    """A neighbourhood of the cells of the next degree that contain a cell.

    On a `CellComplexSpace` a law that reads it sees the faces around a link, or the links that meet at a site, each
    with weight one. Inside a sum over it, `exterior_derivative` of a value on links adds the values around each face
    with their directions, such as the total angle around a square of a lattice. Gauge theories sum the squares around
    each link of a lattice, and flows on a mesh read the faces around each edge.

    Read more in the user guide, under Spaces.

    Args:
        label:
            The name of the neighbourhood, such as `'moore'`.

    See Also:
        AdjacencyNeighborhood: Reads the nodes joined by edges, cells of the same kind.
        CellComplexSpace: Holds the links and faces it reads.

    Examples:
        >>> from skcomplex.core.components.spaces import CofaceNeighborhood
        >>> CofaceNeighborhood(label='plaquettes').prepare().label
        'plaquettes'
    """

    def __init__(self, *, label: str = 'neighborhood') -> None:
        super().__init__(label)
