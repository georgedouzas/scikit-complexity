"""Coordinates and growth rates of Rössler's towel map, a three-dimensional map with two positive exponents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class TowelMapElement(Element):
    """Coordinates and growth rates of Rössler's towel map, a three-dimensional map with two positive exponents.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        a:
            The growth rate of the first logistic coordinate.

        b:
            The growth rate of the third logistic coordinate.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TowelMapElement
        >>> TowelMapElement('map', a=3.7).a
        3.7
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.085,
        y: Annotated[object, StateSpec()] = -0.121,
        z: Annotated[object, StateSpec()] = 0.075,
        a: Annotated[object, ParameterSpec()] = 3.8,
        b: Annotated[object, ParameterSpec()] = 3.78,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
