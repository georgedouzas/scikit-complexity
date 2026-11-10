"""Define the neighbourhood of the members closest to a member."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...errors import ValidationError
from ._base import BaseNeighborhood


class NearestNeighborhood(BaseNeighborhood):
    """A neighbourhood of the `k` members of a continuous space closest to a member.

    Distances are measured between the position states of the members, by the nearest periodic image in a periodic box.
    A group of at most `k` other members gives every member all the others.

    Unlike `RadiusNeighborhood`, which takes every member within a distance, it takes a fixed number of members,
    whatever their distance. Choose it when each member attends to a fixed number of others.

    It appears in many fields. Starlings in a flock follow their seven nearest neighbours, fish school by their closest
    neighbours, and robots of a swarm align with the nearest robots.

    Args:
        k:
            The number of neighbours.

        center:
            Whether a member is its own neighbour, besides its `k` nearest others.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Examples:
        >>> from skcomplex.core.components.spaces import NearestNeighborhood
        >>> NearestNeighborhood(7).k
        7
    """

    def __init__(self, k: int, *, center: bool = False, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.k = k
        self.center = center

    def _prepare(self, context: None = None) -> None:
        """Raise an error for a number of neighbours that is not a whole number above 0.

        The number must be at least 1.
        """
        super()._prepare(context)
        if int(self.k) != self.k or int(self.k) < 1:
            error_msg = f'A nearest neighbourhood needs a whole number of neighbours of at least 1, but got {self.k!r}.'
            raise ValidationError(error_msg)
