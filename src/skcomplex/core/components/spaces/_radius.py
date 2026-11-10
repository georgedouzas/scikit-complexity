"""Define the neighbourhood of the members within a distance, and within a cone."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ...errors import ValidationError
from ._base import BaseNeighborhood


class RadiusNeighborhood(BaseNeighborhood):
    """A neighbourhood of the members of a continuous space within a distance.

    Distances are measured between the position states of the members, by the nearest periodic image in a periodic box.
    With an `angle` below `pi`, a neighbour must also lie in the cone around the direction of the member, so `angle =
    pi` is the whole disk and `angle = pi / 2` the half ahead. The direction is a vector state of the members, or in two
    dimensions a scalar state that holds the angle.

    Unlike `NearestNeighborhood`, which takes a fixed number of members, it takes every member within a distance, so the
    number of neighbours varies. Choose it when members sense others up to a range, optionally within a field of view.

    It appears in many fields. Birds align with flockmates within sight in the Vicsek model, atoms interact within a
    cutoff, and pedestrians react to people ahead of them.

    Args:
        radius:
            The largest distance of a neighbour.

        angle:
            The half-angle of the cone, between 0 and `pi`, which is the whole disk.

        direction:
            The name of the state that holds the direction, or `None` for the velocity, the vector state declared as the
            rate of the position. Only a cone, with an angle below `pi`, reads it.

        center:
            Whether a member is its own neighbour.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Examples:
        >>> import math
        >>> from skcomplex.core.components.spaces import RadiusNeighborhood
        >>> RadiusNeighborhood(1.5).radius
        1.5
        >>> RadiusNeighborhood(1.0, angle=math.pi / 2, direction='theta').direction
        'theta'
    """

    def __init__(
        self,
        radius: float,
        *,
        angle: float = math.pi,
        direction: str | None = None,
        center: bool = False,
        label: str = 'neighborhood',
    ) -> None:
        super().__init__(label)
        self.radius = radius
        self.angle = angle
        self.direction = direction
        self.center = center

    def _prepare(self, context: None = None) -> None:
        """Raise an error for a radius not positive or a half-angle outside `(0, pi]`."""
        super()._prepare(context)
        if not float(self.radius) > 0:
            error_msg = f'A radius neighbourhood needs a positive radius, but got {self.radius!r}.'
            raise ValidationError(error_msg)
        if not 0 < float(self.angle) <= math.pi:
            error_msg = f'A radius neighbourhood needs a half-angle in (0, pi], but got {self.angle!r}.'
            raise ValidationError(error_msg)
