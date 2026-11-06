"""Spherical squirmer, a microswimmer at the position `x` that swims along the unit orientation `p`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec

POINTS = VisualSpec('points', {'position': 'x', 'radius': 1.0})


class SquirmerElement(Element):
    """Spherical squirmer, a microswimmer at the position `x` that swims along the unit orientation `p`.

    The slip of its surface sets how it swims and stirs the fluid, and `SquirmerSlipInteraction` turns the slip of
    every squirmer into the motion of all of them.

    Args:
        label:
            The label of the squirmer or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        p:
            The initial orientation, a unit vector.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import SquirmerElement
        >>> SquirmerElement('swimmer', p=[0.0, 0.0, 1.0]).p
        [0.0, 0.0, 1.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        p: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.p = p
