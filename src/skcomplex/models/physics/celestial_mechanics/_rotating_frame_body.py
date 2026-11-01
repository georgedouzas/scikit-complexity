"""Light body at `(x, y)`, coordinates of order 2 in the frame that rotates with two primaries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


def _primaries(mu: float, **_: float) -> list[list[float]]:
    """Return the positions of the two primaries in the rotating frame."""
    return [[-mu, 0.0], [1 - mu, 0.0]]


class RotatingFrameBodyElement(Element):
    """Light body at `(x, y)`, coordinates of order 2 in the frame that rotates with two primaries.

    The primaries of masses `1 - mu` and `mu` sit at `(-mu, 0)` and `(1 - mu, 0)`.

    Args:
        label:
            The label of the body.

        visual:
            How the body is drawn, or `None` for its class visual.

        x:
            The initial first coordinate.

        x_d1:
            The initial first velocity.

        y:
            The initial second coordinate.

        y_d1:
            The initial second velocity.

        mu:
            The mass of the lighter primary, as a fraction of the total.

    Examples:
        >>> from skcomplex.models.physics.celestial_mechanics import RotatingFrameBodyElement
        >>> RotatingFrameBodyElement('body', mu=0.1).mu
        0.1
    """

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'marks': _primaries})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 0.5,
        x_d1: float = 0.0,
        y: Annotated[float, StateSpec(order=2)] = 0.0,
        y_d1: float = 0.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01215,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.y = y
        self.y_d1 = y_d1
        self.mu = mu
