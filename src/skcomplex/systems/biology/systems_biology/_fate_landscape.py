"""Cell state in the compactified elliptic umbilic, the minimal landscape of a decision among three fates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class FateLandscapeElement(Element):
    """Cell state in the compactified elliptic umbilic, the minimal landscape of a decision among three fates.

    The state `(x, y)` descends the potential `f = x^4 + y^4 + x^3 - 2 x y^2 + a (x^2 + y^2) + theta1 x + theta2 y` of
    Rand et al. (2021, figure 5D), `dx/dt = -df/dx` and `dy/dt = -df/dy`. The signals `theta1` and `theta2` change
    in time at the rates `v1` and `v2`, and the sign of `a` makes the central rest point an attractor or a repellor.

    Args:
        label:
            The label of the cell or of the group of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        a:
            The curvature of the potential at the centre.

        theta1:
            The first signal at the start.

        theta2:
            The second signal at the start.

        v1:
            The rate of change of the first signal.

        v2:
            The rate of change of the second signal.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import FateLandscapeElement
        >>> FateLandscapeElement('cell', a=-0.2).a
        -0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        a: Annotated[float, ParameterSpec()] = -0.2,
        theta1: Annotated[float, ParameterSpec()] = 0.0,
        theta2: Annotated[float, ParameterSpec()] = 0.0,
        v1: Annotated[float, ParameterSpec()] = 0.0,
        v2: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.a = a
        self.theta1 = theta1
        self.theta2 = theta2
        self.v1 = v1
        self.v2 = v2

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the gradient descent of the potential under the drifting signals.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        theta1, theta2 = s.theta1 + s.v1 * s.t, s.theta2 + s.v2 * s.t
        return [
            RateLaw(s.x, -(4 * s.x**3 + 3 * s.x**2 - 2 * s.y**2 + 2 * s.a * s.x + theta1)),
            RateLaw(s.y, -(4 * s.y**3 - 4 * s.x * s.y + 2 * s.a * s.y + theta2)),
        ]
