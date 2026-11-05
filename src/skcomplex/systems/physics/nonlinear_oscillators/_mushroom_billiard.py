"""Free particle in a mushroom table: a half disc cap of radius `radius` on a stem of half width `width`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MushroomBilliardElement(Element):
    """Free particle in a mushroom table: a half disc cap of radius `radius` on a stem of half width `width`.

    The particle flies straight, and reflects specularly off the cap, the floor of the cap beside the stem, the walls of
    the stem, and its floor at depth `height`. Its phase space mixes regular orbits that stay in the cap with chaotic
    orbits that enter the stem.

    Args:
        label:
            The label of the particle.

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        vx:
            The initial first velocity.

        vy:
            The initial second velocity.

        radius:
            The radius of the cap.

        width:
            The half width of the stem.

        height:
            The height of the stem.

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import MushroomBilliardElement
        >>> MushroomBilliardElement('ball').width
        0.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.1,
        y: Annotated[float, StateSpec()] = 0.6,
        vx: Annotated[float, StateSpec()] = 0.6,
        vy: Annotated[float, StateSpec()] = 0.8,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4,
        height: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.width = width
        self.height = height

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the free flight `x' = vx`, `y' = vy` of the particle.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the position.
        """
        s = namespace
        return [RateLaw(s.x, s.vx), RateLaw(s.y, s.vy)]
