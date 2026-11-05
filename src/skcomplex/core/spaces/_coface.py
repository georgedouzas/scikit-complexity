"""Define coface neighbourhoods, the cells of the next degree that contain a cell of a cell complex."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._base import BaseNeighborhood


class CofaceNeighborhood(BaseNeighborhood):
    """Cells of the next degree that contain a cell of a cell complex, the faces of an edge or the edges of a site.

    A reduction over the neighbourhood runs over those cells. Inside it `exterior_derivative` of a state of the reading
    group is the oriented sum of its values around the one coface, which is the holonomy of an abelian connection only,
    such as an angle or an integer modulo a period. Every coface counts with weight one, so the orientation of a coface
    relative to the cell enters through that oriented sum alone.

    Args:
        label:
            The label of the neighbourhood.

    Examples:
        >>> from skcomplex.core.spaces import CofaceNeighborhood
        >>> CofaceNeighborhood().prepare().label
        'neighborhood'
    """

    def __init__(self, *, label: str = 'neighborhood') -> None:
        super().__init__(label)
