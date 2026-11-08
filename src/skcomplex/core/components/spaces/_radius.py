"""Define radius neighbourhoods.

They are the members of a continuous space within a distance.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...errors import ValidationError
from ._base import BaseNeighborhood


class RadiusNeighborhood(BaseNeighborhood):
    """Members of a continuous space within a distance of a member.

    This is the metric neighbourhood. Distances are measured between the position states of the members, by the minimum
    image in a periodic box.

    Args:
        radius:
            The largest distance of a neighbour.

        center:
            Whether a member is its own neighbour.

        label:
            The label of the neighbourhood.

    Examples:
        >>> from skcomplex.core.components.spaces import RadiusNeighborhood
        >>> RadiusNeighborhood(1.5).radius
        1.5
    """

    def __init__(self, radius: float, *, center: bool = False, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.radius = radius
        self.center = center

    def _check(self) -> None:
        """Raise an error for a radius that is not positive."""
        if not float(self.radius) > 0:
            error_msg = f'A radius neighbourhood needs a positive radius, but got {self.radius!r}.'
            raise ValidationError(error_msg)
