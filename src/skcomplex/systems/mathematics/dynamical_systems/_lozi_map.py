"""Coordinates and parameters of the Lozi map `x' = 1 + y - a |x|`, `y' = b x`, which maps iterate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LoziMapElement(Element):
    """Coordinates and parameters of the Lozi map `x' = 1 + y - a |x|`, `y' = b x`, which maps iterate.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        a:
            The slope of the fold.

        b:
            The contraction.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LoziMapElement
        >>> LoziMapElement('map', a=1.5).a
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        y: Annotated[object, StateSpec()] = 0.0,
        a: Annotated[object, ParameterSpec()] = 1.7,
        b: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.a = a
        self.b = b
