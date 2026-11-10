"""Define the neighbourhood of the cells of the next degree that contain a cell."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseNeighborhood


class CofaceNeighborhood(BaseNeighborhood):
    """A neighbourhood of the cells of the next degree that contain a cell.

    The cofaces of an edge are the faces it bounds, and the cofaces of a site are the edges that meet at it. A reduction
    over the neighbourhood runs over those cells. Inside it, `exterior_derivative` of a state of the reading group is
    the oriented sum of its values around the one coface, the holonomy of a connection whose values are angles or
    integers modulo a period. Every coface counts with weight one.

    Unlike `AdjacencyNeighborhood`, which reads nodes of the same kind, it reads cells of the next degree. Choose it
    when a state on links depends on the faces around each link.

    It appears in physics and numerical analysis. Lattice gauge theories sum the plaquettes around each link, and
    discrete exterior calculus reads the faces around each edge of a mesh.

    Args:
        label:
            The name of the neighbourhood, such as `'moore'`.

    Examples:
        >>> from skcomplex.core.components.spaces import CofaceNeighborhood
        >>> CofaceNeighborhood().prepare().label
        'neighborhood'
    """

    def __init__(self, *, label: str = 'neighborhood') -> None:
        super().__init__(label)
