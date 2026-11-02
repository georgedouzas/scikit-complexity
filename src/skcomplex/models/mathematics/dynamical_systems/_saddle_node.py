"""Normal form `x' = mu - x^2` of the saddle-node bifurcation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class SaddleNodeElement(Element):
    """Normal form `x' = mu - x^2` of the saddle-node bifurcation.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        mu:
            The bifurcation parameter.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import SaddleNodeElement
        >>> SaddleNodeElement('form', mu=-1.0).mu
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        mu: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.mu = mu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.mu - s.x**2)]
