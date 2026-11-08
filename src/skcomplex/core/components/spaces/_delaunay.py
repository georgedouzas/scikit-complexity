"""Define Delaunay neighbourhoods.

They are the members of a continuous space joined to a member by an edge of the Delaunay triangulation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseNeighborhood


class DelaunayNeighborhood(BaseNeighborhood):
    """Members of a continuous space joined to a member by an edge of the Delaunay triangulation of the positions.

    It is the natural neighbourhood of points in the plane or in space, the members whose Voronoi cells touch, as in
    cells joined by springs, foams, territories, and meshes of moving particles. The triangulation is computed again
    at every step, so the neighbours follow moving members, and in a periodic box it covers the periodic images, so
    neighbours connect across the boundary. A group with too few members to triangulate, or members in a degenerate
    position, gives every member all the others.

    Args:
        center:
            Whether a member is its own neighbour.

        label:
            The label of the neighbourhood.

    Examples:
        >>> from skcomplex.core.components.spaces import DelaunayNeighborhood
        >>> DelaunayNeighborhood(center=True).center
        True
    """

    def __init__(self, *, center: bool = False, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.center = center
