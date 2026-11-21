"""Define the neighbourhood of the members that touch, with no member between them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseNeighborhood


class DelaunayNeighborhood(BaseNeighborhood):
    """A neighbourhood of the members joined by edges of a Delaunay triangulation.

    In a continuous space two members are neighbours when the regions of the plane closer to each of them than to any
    other member touch, whatever their distance. The neighbours are found again at every step, so they follow moving
    members, and in a periodic box they connect across the walls. A group too small to triangulate gives every member
    all the others. Cells in a tissue push on the cells that touch them, bubbles share walls in a foam, and animals
    guard their territories against the animals next door.

    Read more in the user guide, under Spaces.

    Args:
        center:
            Whether a member is its own neighbour.

        label:
            The name of the neighbourhood, such as `'moore'`.

    See Also:
        RadiusNeighborhood: Picks every member within a distance.
        NearestNeighborhood: Picks a fixed number of the closest members.

    Examples:
        >>> from skcomplex.core.components.spaces import DelaunayNeighborhood
        >>> DelaunayNeighborhood(center=True).prepare().center
        True
    """

    def __init__(self, *, center: bool = False, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.center = center
