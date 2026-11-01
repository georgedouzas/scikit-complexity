"""Rössler oscillator `x1' = -omega x2 - x3`, `x2' = omega x1 + a x2`, `x3' = b + x3 (x1 - c)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RosslerOscillatorElement(Element):
    """Rössler oscillator `x1' = -omega x2 - x3`, `x2' = omega x1 + a x2`, `x3' = b + x3 (x1 - c)`.

    It is chaotic for `a = b = 0.2` and `c = 5.7`.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x1:
            The initial first variable.

        x2:
            The initial second variable.

        x3:
            The initial third variable.

        omega:
            The natural frequency.

        a:
            The first Rössler parameter.

        b:
            The second Rössler parameter.

        c:
            The third Rössler parameter.

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.models.physics.synchronization import RosslerOscillatorElement
        >>> RosslerOscillatorElement('oscillator').c
        5.7
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x1: Annotated[float, StateSpec()] = 1.0,
        x2: Annotated[float, StateSpec()] = 0.0,
        x3: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
        a: Annotated[float, ParameterSpec()] = 0.2,
        b: Annotated[float, ParameterSpec()] = 0.2,
        c: Annotated[float, ParameterSpec()] = 5.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.omega = omega
        self.a = a
        self.b = b
        self.c = c

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the Rössler flow.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the three variables.
        """
        s = namespace
        return [
            RateLaw(s.x1, -s.omega * s.x2 - s.x3),
            RateLaw(s.x2, s.omega * s.x1 + s.a * s.x2),
            RateLaw(s.x3, s.b + s.x3 * (s.x1 - s.c)),
        ]
