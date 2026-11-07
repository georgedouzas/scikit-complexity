"""Sprott's flow `x' = y + 2xy + xz`, `y' = 1 - 2x² + yz`, `z' = x - x² - y²`, conservative and dissipative at once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import StateSpec, VisualSpec


class MixedFlowElement(Element):
    """Sprott's flow `x' = y + 2xy + xz`, `y' = 1 - 2x² + yz`, `z' = x - x² - y²`, conservative and dissipative at once.

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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import MixedFlowElement
        >>> MixedFlowElement('flow', x=2.0).x
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
        y: Annotated[object, StateSpec()] = 0.0,
        z: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z

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
            RateLaw(s.x, s.y + 2 * s.x * s.y + s.x * s.z),
            RateLaw(s.y, 1 - 2 * s.x**2 + s.y * s.z),
            RateLaw(s.z, s.x - s.x**2 - s.y**2),
        ]
