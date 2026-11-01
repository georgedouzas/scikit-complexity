"""Planet at `(x, y)` relative to a heavy star, with the canonical momenta `(px, py)` of the Kepler problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


def _star(**_: float) -> list[list[float]]:
    """Return the position of the star at the origin of the relative coordinate."""
    return [[0.0, 0.0]]


class KeplerBodyElement(Element):
    """Planet at `(x, y)` relative to a heavy star, with the canonical momenta `(px, py)` of the Kepler problem.

    The states are of order 1, and the star of mass `mass` sits at the origin.

    Args:
        label:
            The label of the planet.

        visual:
            How the planet is drawn, or `None` for its class visual.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        px:
            The initial first momentum per unit mass.

        py:
            The initial second momentum per unit mass.

        G:
            The gravitational constant.

        mass:
            The mass of the star.

    Examples:
        >>> from skcomplex.models.physics.celestial_mechanics import KeplerBodyElement
        >>> KeplerBodyElement('planet', mass=10.0).mass
        10.0
    """

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'marks': _star})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 10.0,
        y: Annotated[float, StateSpec()] = 0.0,
        px: Annotated[float, StateSpec()] = 0.0,
        py: Annotated[float, StateSpec()] = 10.0,
        G: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        mass: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.G = G
        self.mass = mass
