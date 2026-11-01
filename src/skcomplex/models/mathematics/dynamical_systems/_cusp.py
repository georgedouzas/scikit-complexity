"""Normal form `x' = beta1 + beta2 x - x^3` of the cusp bifurcation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class CuspElement(Element):
    """Normal form `x' = beta1 + beta2 x - x^3` of the cusp bifurcation.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        beta1:
            The first unfolding parameter, the asymmetry.

        beta2:
            The second unfolding parameter, the splitting.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import CuspElement
        >>> CuspElement('form', beta2=2.0).beta2
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.5,
        beta1: Annotated[object, ParameterSpec()] = 0.0,
        beta2: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.beta1 = beta1
        self.beta2 = beta2

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.beta1 + s.beta2 * s.x - s.x**3)]
