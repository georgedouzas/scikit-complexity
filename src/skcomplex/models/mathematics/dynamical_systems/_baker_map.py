"""Coordinates and parameters of the generalized baker map, which stretches the unit square and stacks it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class BakerMapElement(Element):
    """Coordinates and parameters of the generalized baker map, which stretches the unit square and stacks it.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial stretched coordinate.

        y:
            The initial squeezed coordinate.

        alpha:
            The position of the cut.

        lambda_a:
            The height of the bottom band.

        lambda_b:
            The height of the top band.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import BakerMapElement
        >>> BakerMapElement('map', alpha=0.4).alpha
        0.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.2,
        y: Annotated[object, StateSpec()] = 0.3,
        alpha: Annotated[object, ParameterSpec()] = 1 / 3,
        lambda_a: Annotated[object, ParameterSpec()] = 1 / 3,
        lambda_b: Annotated[object, ParameterSpec()] = 2 / 3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
