"""Colloidal sphere at the position `x`, held by an optical trap centred at `center`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

SPHERES = VisualSpec('points', {'position': 'x', 'radius': 1.0})


class TrappedSphereElement(Element):
    """Colloidal sphere at the position `x`, held by an optical trap centred at `center`.

    The sphere has no friction of its own. A mobility interaction moves it through the fluid it shares with the other
    spheres, and its trap is one of the forces the mobility turns into velocities.

    Args:
        label:
            The label of the sphere or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        center:
            The centre of the trap of the sphere.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import TrappedSphereElement
        >>> TrappedSphereElement('bead', center=[3.0, 0.0]).center
        [3.0, 0.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = SPHERES,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        center: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.center = center
