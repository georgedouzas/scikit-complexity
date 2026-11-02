"""Particle moving freely in the plane with the position `(x, y)` and the velocity `(vx, vy)`, inside a disc."""

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

WALL_POINTS = 200


def _disc_wall(radius: float, **_: float) -> list[np.ndarray]:
    """Return the wall of a disc table as one closed polyline."""
    angles = np.linspace(0.0, 2 * np.pi, WALL_POINTS)
    return [radius * np.column_stack([np.cos(angles), np.sin(angles)])]


class DiscBilliardElement(Element):
    """Particle moving freely in the plane with the position `(x, y)` and the velocity `(vx, vy)`, inside a disc.

    The position changes at the velocity, which changes only at the reflections off the wall of the disc.

    Args:
        label:
            The label of the particle.

        visual:
            How the particle is drawn, or `None` for its class visual.

        x:
            The initial horizontal position.

        y:
            The initial vertical position.

        vx:
            The initial horizontal velocity.

        vy:
            The initial vertical velocity.

        radius:
            The radius of the disc.

    Examples:
        >>> from skcomplex.models.physics.mechanics import DiscBilliardElement
        >>> DiscBilliardElement('particle', radius=2.0).radius
        2.0
    """

    visual = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None, 'outline': _disc_wall})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.3,
        y: Annotated[float, StateSpec()] = 0.0,
        vx: Annotated[float, StateSpec()] = 0.4,
        vy: Annotated[float, StateSpec()] = 1.0,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the free flight `x' = vx`, `y' = vy` of the particle.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the position.
        """
        return [RateLaw(s.x, s.vx), RateLaw(s.y, s.vy)]
