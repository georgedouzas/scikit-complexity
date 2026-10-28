"""Define nearest neighbourhoods, the members of a continuous space closest to a member."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT
from typing import Self

from ...errors import ValidationError
from ._base import BaseNeighborhood


class NearestNeighborhood(BaseNeighborhood):
    """The `k` members of a continuous space closest to a member, the topological neighbourhood.

    Distances are measured between the position states of the members, by the minimum image in a periodic box. A group
    of at most `k` other members gives every member all the others.

    Args:
        k:
            The number of neighbours.

        center:
            Whether a member is its own neighbour, besides its `k` nearest others.

        label:
            The label of the neighbourhood.

    Examples:
        >>> from skcomplex.core.spaces import NearestNeighborhood
        >>> NearestNeighborhood(7).check().k
        7
    """

    def __init__(self, k: int, *, center: bool = False, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.k = k
        self.center = center

    def _check(self) -> Self:
        """Raise an error for a number of neighbours that is not a whole number of at least 1."""
        if int(self.k) != self.k or int(self.k) < 1:
            error_msg = f'A nearest neighbourhood needs a whole number of neighbours of at least 1, but got {self.k!r}.'
            raise ValidationError(error_msg)
        return self
