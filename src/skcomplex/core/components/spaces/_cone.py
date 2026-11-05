"""Define cone neighbourhoods, the members of a continuous space within a distance and a cone around a direction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ...errors import ValidationError
from ._base import BaseNeighborhood


class ConeNeighborhood(BaseNeighborhood):
    """Members of a continuous space within a distance of a member and inside a cone around its direction.

    A neighbour is within the radius of a member and its direction from the member makes an angle of at most `angle`
    with the member's direction, so `angle = pi` is the whole disk and `angle = pi / 2` the half ahead. The direction is
    a vector state of the members, or in two dimensions a scalar state that holds the direction angle.

    Args:
        radius:
            The largest distance of a neighbour.

        angle:
            The half-angle of the cone, between 0 and `pi`.

        direction:
            The name of the state that holds the direction, or `None` for the velocity, the vector state declared as the
            rate of the position.

        center:
            Whether a member is its own neighbour.

        label:
            The label of the neighbourhood.

    Examples:
        >>> import math
        >>> from skcomplex.core.components.spaces import ConeNeighborhood
        >>> ConeNeighborhood(1.0, math.pi / 2, direction='theta').direction
        'theta'
    """

    def __init__(
        self,
        radius: float,
        angle: float,
        *,
        direction: str | None = None,
        center: bool = False,
        label: str = 'neighborhood',
    ) -> None:
        super().__init__(label)
        self.radius = radius
        self.angle = angle
        self.direction = direction
        self.center = center

    def _check(self) -> None:
        """Raise an error for a radius that is not positive or a half-angle outside `(0, pi]`."""
        if not float(self.radius) > 0:
            error_msg = f'A cone neighbourhood needs a positive radius, but got {self.radius!r}.'
            raise ValidationError(error_msg)
        if not 0 < float(self.angle) <= math.pi:
            error_msg = f'A cone neighbourhood needs a half-angle in (0, pi], but got {self.angle!r}.'
            raise ValidationError(error_msg)
