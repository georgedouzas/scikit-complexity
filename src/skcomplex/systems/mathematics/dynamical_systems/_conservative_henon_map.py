"""Coordinates and rotation of Hénon's area-preserving map, a rotation by `2π nu` followed by a quadratic kick."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ConservativeHenonMapElement(Element):
    """Coordinates and rotation of Hénon's area-preserving map, a rotation by `2π nu` followed by a quadratic kick.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        nu:
            The rotation number of the linear part, in turns.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ConservativeHenonMapElement
        >>> ConservativeHenonMapElement('map', nu=0.25).nu
        0.25
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        nu: Annotated[object, ParameterSpec()] = 0.2114,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.nu = nu
