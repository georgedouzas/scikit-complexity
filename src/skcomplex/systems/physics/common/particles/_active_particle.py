"""Active particle in the plane.

The particle has a position `x` and an orientation `theta`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

from .....core.components.elements import Element, Period
from .....core.components.spaces import BaseSpace
from .....core.specs import VisualSpec

POINTS = VisualSpec('points', {'position': 'x', 'radius': 0.2})


class ActiveParticleElement(Element):
    """Active particle in the plane with the position `x` and the orientation `theta`.

    A `SelfPropulsionInteraction` moves the position along the orientation, and an `ActiveDiffusionInteraction` makes
    both diffuse and the orientation turn.

    Args:
        label:
            The label of the particle or the group.

        x:
            The initial position, one row per particle or shared.

        theta:
            The initial orientation, one per particle or shared.

        size:
            The number of particles, or `None` for one.

        space:
            The continuous space of the particles, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import ActiveParticleElement
        >>> ActiveParticleElement('swimmer', theta=1.0).theta
        1.0
    """

    visual = POINTS

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.theta = theta
