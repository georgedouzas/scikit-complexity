"""Free particle scattered by three discs of radius `radius` at the corners of a triangle of side `spacing`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

ANGLES = (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)
"""The polar angles of the centers of the three discs, one below the origin and two above it."""


def _three_disk_walls(spacing: float, radius: float, **_: float) -> list[np.ndarray]:
    """Return the three discs of an open three-disk billiard as closed polylines."""
    angles = np.linspace(0.0, 2 * np.pi, 100)
    circle = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    distance = spacing / math.sqrt(3)
    return [circle + distance * np.array([math.cos(angle), math.sin(angle)]) for angle in ANGLES]


class ThreeDiskBilliardElement(Element):
    """Free particle scattered by three discs of radius `radius` at the corners of a triangle of side `spacing`.

    The centers of the discs lie at the distance `spacing / √3` from the origin, at the polar angles `-π / 2`,
    `π / 6`, and `5 π / 6`. The particle flies straight and reflects specularly off the discs. The billiard is open,
    so almost every particle leaves the scattering region after a finite number of collisions, and the number of
    collisions depends on the incoming direction in a fractal way.

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

        spacing:
            The distance between the centers of two discs.

        radius:
            The radius of the discs.

    Examples:
        >>> from skcomplex.systems.physics.nonlinear_oscillators import ThreeDiskBilliardElement
        >>> ThreeDiskBilliardElement('ball').spacing
        2.4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec(
            'bodies',
            {'positions': ('x', 'y'), 'trail': None, 'outline': _three_disk_walls},
        ),
        *,
        x: Annotated[float, StateSpec()] = 0.1,
        y: Annotated[float, StateSpec()] = 0.6,
        vx: Annotated[float, StateSpec()] = 0.6,
        vy: Annotated[float, StateSpec()] = 0.8,
        spacing: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.4,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.4,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.spacing = spacing
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
