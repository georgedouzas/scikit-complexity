"""Star in the Hénon-Heiles potential `V = (x^2 + y^2) / 2 + x^2 y - y^3 / 3`, with `x'' = -dV/dx`, `y'' = -dV/dy`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import StateSpec, VisualSpec


class HenonHeilesElement(Element):
    """Star in the Hénon-Heiles potential `V = (x^2 + y^2) / 2 + x^2 y - y^3 / 3`, with `x'' = -dV/dx`, `y'' = -dV/dy`.

    Below the escape energy `1 / 6` the motion is regular for small energies and mostly chaotic near the escape energy.

    Args:
        label:
            The label of the star.

        visual:
            How the star is drawn.

        x:
            The initial first coordinate.

        x_d1:
            The initial first momentum.

        y:
            The initial second coordinate.

        y_d1:
            The initial second momentum.

    Attributes:
        default_visual_:
            The visual of the star when it sets none.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import HenonHeilesElement
        >>> HenonHeilesElement('star').x_d1
        0.35
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None}),
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.0,
        x_d1: float = 0.35,
        y: Annotated[float, StateSpec(order=2)] = 0.1,
        y_d1: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.y = y
        self.y_d1 = y_d1

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the accelerations of the two coordinates.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the second derivatives of `x` and `y`.
        """
        s = namespace
        return [RateLaw(s.x, -s.x - 2 * s.x * s.y), RateLaw(s.y, -s.y - s.x**2 + s.y**2)]
