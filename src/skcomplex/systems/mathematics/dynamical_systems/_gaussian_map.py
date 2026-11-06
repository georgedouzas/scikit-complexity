"""Value and parameters of the CDN Gaussian map `x' = exp(-alpha² x²) + β` of an interval."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class GaussianMapElement(Element):
    """Value and parameters of the CDN Gaussian map `x' = exp(-alpha² x²) + β` of an interval.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        alpha:
            The steepness of the Gaussian.

        beta:
            The shift of the Gaussian.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import GaussianMapElement
        >>> GaussianMapElement('map', beta=-0.3).beta
        -0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        alpha: Annotated[object, ParameterSpec()] = 2.8,
        beta: Annotated[object, ParameterSpec()] = -0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.alpha = alpha
        self.beta = beta
