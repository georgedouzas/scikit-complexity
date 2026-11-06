"""Light body at `(x, y)` with the canonical momenta `(px, py)` in the frame that rotates with two primaries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


def _primaries(mu: float, **_: float) -> list[list[float]]:
    """Return the positions of the two primaries in the rotating frame."""
    return [[-mu, 0.0], [1 - mu, 0.0]]


class RotatingFrameBodyElement(Element):
    """Light body at `(x, y)` with the canonical momenta `(px, py)` in the frame that rotates with two primaries.

    The states are of order 1, and the momenta are `px = x' - y` and `py = y' + x` for the unit angular velocity of the
    frame. The primaries of masses `1 - mu` and `mu` sit at `(-mu, 0)` and `(1 - mu, 0)`.

    Args:
        label:
            The label of the body.

        visual:
            How the body is drawn, or `None` for its class visual.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        px:
            The initial first momentum.

        py:
            The initial second momentum.

        mu:
            The mass of the lighter primary, as a fraction of the total.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import RotatingFrameBodyElement
        >>> RotatingFrameBodyElement('body', mu=0.1).mu
        0.1
    """

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'marks': _primaries})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.5,
        y: Annotated[float, StateSpec()] = 0.0,
        px: Annotated[float, StateSpec()] = 0.0,
        py: Annotated[float, StateSpec()] = 0.5,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01215,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.mu = mu
