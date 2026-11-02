"""Free particle in a square table of half side `size` with a central disc of radius `radius`, the Sinai billiard."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


def _sinai_walls(size: float, radius: float, **_: float) -> list[np.ndarray]:
    """Return the square wall and the central disc of a Sinai table as closed polylines."""
    square = size * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    angles = np.linspace(0.0, 2 * np.pi, 100)
    return [square, radius * np.column_stack([np.cos(angles), np.sin(angles)])]


class SinaiBilliardElement(Element):
    """Free particle in a square table of half side `size` with a central disc of radius `radius`, the Sinai billiard.

    The particle flies straight, and its specular reflections off the four walls and the disc make its motion ergodic
    and chaotic.

    Args:
        label:
            The label of the particle.

        visual:
            How the particle is drawn.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        vx:
            The initial first velocity.

        vy:
            The initial second velocity.

        size:
            The half side of the square.

        radius:
            The radius of the disc.

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import SinaiBilliardElement
        >>> SinaiBilliardElement('ball').radius
        0.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec(
            'bodies',
            {'positions': ('x', 'y'), 'trail': None, 'outline': _sinai_walls},
        ),
        *,
        x: Annotated[float, StateSpec()] = 0.1,
        y: Annotated[float, StateSpec()] = 0.6,
        vx: Annotated[float, StateSpec()] = 0.6,
        vy: Annotated[float, StateSpec()] = 0.8,
        size: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.radius = radius

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
