"""Coordinates and parameters of the Hénon map `x' = 1 - a x^2 + y`, `y' = b x`, which maps iterate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class HenonMapElement(Element):
    """Coordinates and parameters of the Hénon map `x' = 1 - a x^2 + y`, `y' = b x`, which maps iterate.

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
            The nonlinearity.

        b:
            The contraction.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import HenonMapElement
        >>> HenonMapElement('map', a=1.2).a
        1.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        y: Annotated[object, StateSpec()] = 0.0,
        a: Annotated[object, ParameterSpec()] = 1.4,
        b: Annotated[object, ParameterSpec()] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.a = a
        self.b = b
