"""Rössler flow `x' = -y - z`, `y' = x + a y`, `z' = b + z (x - c)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RosslerElement(Element):
    """Rössler flow `x' = -y - z`, `y' = x + a y`, `z' = b + z (x - c)`.

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

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The threshold of the third coordinate's growth.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import RosslerElement
        >>> RosslerElement('rossler', c=4.0).c
        4.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 1.0,
        z: Annotated[object, StateSpec()] = 1.0,
        a: Annotated[object, ParameterSpec()] = 0.2,
        b: Annotated[object, ParameterSpec()] = 0.2,
        c: Annotated[object, ParameterSpec()] = 5.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the three coordinates.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            RateLaw(s.x, -s.y - s.z),
            RateLaw(s.y, s.x + s.a * s.y),
            RateLaw(s.z, s.b + s.z * (s.x - s.c)),
        ]
