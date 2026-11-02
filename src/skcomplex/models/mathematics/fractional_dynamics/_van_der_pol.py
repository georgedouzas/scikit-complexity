"""Define the Van der Pol oscillator, a relaxation oscillator with nonlinear damping."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class VanDerPolElement(Element):
    """Van der Pol oscillator `x' = y`, `y' = mu (1 - x^2) y - x` of two first-order states.

    Its fractional subclasses replace both derivatives by Caputo derivatives of one order.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        y:
            The initial derivative of the displacement.

        mu:
            The strength of the nonlinear damping.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import VanDerPolElement
        >>> VanDerPolElement('oscillator', mu=2.0).mu
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 2.0,
        y: Annotated[float, StateSpec()] = 0.0,
        mu: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.mu = mu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the displacement and its derivative.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.y), RateLaw(s.y, s.mu * (1 - s.x**2) * s.y - s.x)]
