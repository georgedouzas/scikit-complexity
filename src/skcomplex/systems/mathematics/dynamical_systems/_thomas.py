"""Thomas flow `x' = -a x + sin y`, `y' = -a y + sin z`, `z' = -a z + sin x`, cyclically symmetric and damped."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import sin

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ThomasElement(Element):
    """Thomas flow `x' = -a x + sin y`, `y' = -a y + sin z`, `z' = -a z + sin x`, cyclically symmetric and damped.

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
            The damping of every coordinate.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ThomasElement
        >>> ThomasElement('thomas', a=0.2).a
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        z: Annotated[object, StateSpec()] = 0.0,
        a: Annotated[object, ParameterSpec()] = 0.18,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.a = a

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
            RateLaw(s.x, -s.a * s.x + sin(s.y)),
            RateLaw(s.y, -s.a * s.y + sin(s.z)),
            RateLaw(s.z, -s.a * s.z + sin(s.x)),
        ]
