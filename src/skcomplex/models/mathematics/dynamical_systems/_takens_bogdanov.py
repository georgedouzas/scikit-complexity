"""Normal form `x' = y`, `y' = beta1 + beta2 x + x^2 + sign x y` of the Takens-Bogdanov bifurcation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class TakensBogdanovElement(Element):
    """Normal form `x' = y`, `y' = beta1 + beta2 x + x^2 + sign x y` of the Takens-Bogdanov bifurcation.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        beta1:
            The first unfolding parameter.

        beta2:
            The second unfolding parameter.

        sign:
            The sign of the term `x y`, 1 or -1.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import TakensBogdanovElement
        >>> TakensBogdanovElement('form', beta1=0.0).beta1
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.2,
        y: Annotated[object, StateSpec()] = 0.0,
        beta1: Annotated[object, ParameterSpec()] = 0.1,
        beta2: Annotated[object, ParameterSpec()] = -1.0,
        sign: Annotated[object, ParameterSpec()] = -1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.beta1 = beta1
        self.beta2 = beta2
        self.sign = sign

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the two coordinates.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.y), RateLaw(s.y, s.beta1 + s.beta2 * s.x + s.x**2 + s.sign * s.x * s.y)]
