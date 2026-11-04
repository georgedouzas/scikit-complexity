"""Boid in three dimensions, a bird that steers by the velocity and the position of the boids it sees."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._spatial_boid_view import SpatialBoidView


class SpatialBoidElement(Element):
    """Boid in three dimensions, a bird that steers by the velocity and the position of the boids it sees.

    Every step `HeadingInteraction` steers a boid by cohesion, separation, and alignment with the boids within `radius`,
    with its speed between `min_speed` and `max_speed`, and `SpatialFlightInteraction` moves it along its velocity.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        velocity:
            The initial velocity, one component per dimension of the space.

        radius:
            The distance within which a boid sees others.

        separation_radius:
            The distance within which a boid avoids others.

        cohesion:
            The weight of cohesion.

        separation:
            The weight of separation.

        alignment:
            The weight of alignment.

        min_speed:
            The smallest speed.

        max_speed:
            The largest speed.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.collective_behaviour import SpatialBoidElement
        >>> GroupElement('flock', element=SpatialBoidElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents'),
        *,
        velocity: Annotated[npt.ArrayLike, StateSpec(vector=True, role='velocity')] = 0.0,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        separation_radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        cohesion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        separation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        alignment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        min_speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        max_speed: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.velocity = velocity
        self.radius = radius
        self.separation_radius = separation_radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.min_speed = min_speed
        self.max_speed = max_speed

    @classmethod
    def _get_view(cls) -> type[SpatialBoidView]:
        """Return the view of a boid, `SpatialBoidView`."""
        return SpatialBoidView
