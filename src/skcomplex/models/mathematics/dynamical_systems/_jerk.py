"""Sprott's jerk flow `x''' = -A x'' - x' + |x| - 1` of a state of order 3."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import abs_symbolic

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative
from ....specs import ParameterSpec, StateSpec, VisualSpec


class JerkElement(Element):
    """Sprott's jerk flow `x''' = -A x'' - x' + |x| - 1` of a state of order 3.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        x_d2:
            The initial second derivative.

        x_d1:
            The initial first derivative.

        A:
            The damping of the second derivative.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import JerkElement
        >>> JerkElement('jerk', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=3, initial=(0.0, 1.0))] = 0.0,
        x_d2: float = 1.0,
        x_d1: float = 0.0,
        A: Annotated[object, ParameterSpec()] = 0.6,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d2 = x_d2
        self.x_d1 = x_d1
        self.A = A

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the third derivative of the position.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, -s.A * derivative(s.x, 2) - derivative(s.x, 1) + abs_symbolic(s.x) - 1)]
