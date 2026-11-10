"""Define the neighbourhood of members joined by edges of a Delaunay triangulation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseNeighborhood


class DelaunayNeighborhood(BaseNeighborhood):
    """A neighbourhood of the members joined by edges of a Delaunay triangulation.

    The Delaunay triangulation of a set of points joins two points when their Voronoi cells, the regions closer to them
    than to any other point, touch. The triangulation is of the positions of the members of a continuous space, and is
    computed again at every step, so the neighbours follow moving members. In a periodic box it covers the periodic
    images, so neighbours connect across the boundary. A group with too few members to triangulate, or in a degenerate
    position, gives every member all the others.

    Unlike `RadiusNeighborhood`, which takes every member within a distance, it takes the members that touch, whatever
    the distance. Choose it when members interact with those beside them, with no gap between.

    It appears in many fields. Cells in a tissue push on the cells that touch them, bubbles share walls in a foam, and
    animals defend territories against their neighbours.

    Args:
        center:
            Whether a member is its own neighbour.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Examples:
        >>> from skcomplex.core.components.spaces import DelaunayNeighborhood
        >>> DelaunayNeighborhood(center=True).center
        True
    """

    def __init__(self, *, center: bool = False, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.center = center
