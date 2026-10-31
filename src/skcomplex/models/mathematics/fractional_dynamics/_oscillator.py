"""Define the damped linear oscillator as a pair of first-order states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class OscillatorElement(Element):
    """Damped linear oscillator `x' = v`, `v' = -omega^2 x - 2 gamma v` of two first-order states.

    Its fractional subclasses replace both derivatives by Caputo derivatives of one order.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        v:
            The initial derivative of the displacement.

        omega:
            The angular frequency.

        gamma:
            The damping rate.

    Examples:
        >>> from skcomplex.models.mathematics.fractional_dynamics import OscillatorElement
        >>> OscillatorElement('oscillator', omega=2.0).omega
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        v: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
        gamma: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.omega = omega
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the displacement and its derivative.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.v), RateLaw(s.v, -(s.omega**2) * s.x - 2 * s.gamma * s.v)]
