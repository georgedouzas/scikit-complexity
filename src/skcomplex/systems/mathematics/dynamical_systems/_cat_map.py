"""Coordinates of Arnold's cat map `(x, y) -> (2 x + y, x + y) (mod 1)` of the unit torus, which maps iterate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class CatMapElement(Element):
    """Coordinates of Arnold's cat map `(x, y) -> (2 x + y, x + y) (mod 1)` of the unit torus, which maps iterate.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate, between 0 and 1.

        y:
            The initial second coordinate, between 0 and 1.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CatMapElement
        >>> CatMapElement('map', x=0.5).x
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
