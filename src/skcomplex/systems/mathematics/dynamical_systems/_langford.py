"""Langford flow, a rotation in `x, y` whose radius and height `z` couple into a torus that breaks into chaos."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class LangfordElement(Element):
    """Langford flow, a rotation in `x, y` whose radius and height `z` couple into a torus that breaks into chaos.

    The flow is `x' = (z - b) x - d y`, `y' = d x + (z - b) y`, and
    `z' = c + a z - z³ / 3 - (x² + y²)(1 + e z) + f z x³`.

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
            The initial height.

        a:
            The linear growth of the height.

        b:
            The height above which the rotation grows.

        c:
            The constant drive of the height.

        d:
            The angular frequency of the rotation.

        e:
            The coupling of the radius and the height.

        f:
            The term that breaks the rotational symmetry.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import LangfordElement
        >>> LangfordElement('langford', d=3.0).d
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        z: Annotated[object, StateSpec()] = 0.0,
        a: Annotated[object, ParameterSpec()] = 0.95,
        b: Annotated[object, ParameterSpec()] = 0.7,
        c: Annotated[object, ParameterSpec()] = 0.6,
        d: Annotated[object, ParameterSpec()] = 3.5,
        e: Annotated[object, ParameterSpec()] = 0.25,
        f: Annotated[object, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the three coordinates.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        radius = s.x**2 + s.y**2
        return [
            RateLaw(s.x, (s.z - s.b) * s.x - s.d * s.y),
            RateLaw(s.y, s.d * s.x + (s.z - s.b) * s.y),
            RateLaw(s.z, s.c + s.a * s.z - s.z**3 / 3 - radius * (1 + s.e * s.z) + s.f * s.z * s.x**3),
        ]
