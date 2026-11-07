"""Rössler's four-dimensional flow `x' = -(y + z)`, `y' = x + a y + w`, `z' = b + x z`, `w' = c w - d z`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class HyperchaoticRosslerElement(Element):
    """Rössler's four-dimensional flow `x' = -(y + z)`, `y' = x + a y + w`, `z' = b + x z`, `w' = c w - d z`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        w:
            The initial fourth coordinate.

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The growth of the fourth coordinate.

        d:
            The damping of the fourth coordinate by the third.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HyperchaoticRosslerElement
        >>> HyperchaoticRosslerElement('rossler', b=2.0).b
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = -10.0,
        y: Annotated[object, StateSpec()] = -6.0,
        z: Annotated[object, StateSpec()] = 0.0,
        w: Annotated[object, StateSpec()] = 10.0,
        a: Annotated[object, ParameterSpec()] = 0.25,
        b: Annotated[object, ParameterSpec()] = 3.0,
        c: Annotated[object, ParameterSpec()] = 0.05,
        d: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the four coordinates.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            RateLaw(s.x, -(s.y + s.z)),
            RateLaw(s.y, s.x + s.a * s.y + s.w),
            RateLaw(s.z, s.b + s.x * s.z),
            RateLaw(s.w, s.c * s.w - s.d * s.z),
        ]
