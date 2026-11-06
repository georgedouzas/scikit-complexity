"""Hyperbolic growth `x' = a x^p`, which diverges in finite time for `p > 1`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class BlowUpElement(Element):
    """Hyperbolic growth `x' = a x^p`, which diverges in finite time for `p > 1`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        a:
            The growth coefficient.

        p:
            The growth exponent.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import BlowUpElement
        >>> BlowUpElement('growth', p=3.0).p
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        a: Annotated[object, ParameterSpec()] = 1.0,
        p: Annotated[object, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a
        self.p = p

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.a * s.x**s.p)]
