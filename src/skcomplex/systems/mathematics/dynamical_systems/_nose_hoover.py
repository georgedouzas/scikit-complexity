"""Nosé-Hoover oscillator `x' = y`, `y' = y z - x`, `z' = 1 - y²`, a conservative flow with tori and chaos."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import StateSpec, VisualSpec


class NoseHooverElement(Element):
    """Nosé-Hoover oscillator `x' = y`, `y' = y z - x`, `z' = 1 - y²`, a conservative flow with tori and chaos.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        y:
            The initial momentum.

        z:
            The initial friction of the thermostat.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import NoseHooverElement
        >>> NoseHooverElement('oscillator', y=1.0).y
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        y: Annotated[object, StateSpec()] = 0.1,
        z: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the position, the momentum, and the friction.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.y), RateLaw(s.y, s.y * s.z - s.x), RateLaw(s.z, 1 - s.y**2)]
