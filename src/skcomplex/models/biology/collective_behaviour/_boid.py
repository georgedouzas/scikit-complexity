"""Boid of Reynolds, a bird that steers by three rules toward, away from, and along its neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._boid_view import BoidView


class BoidElement(Element):
    """Boid of Reynolds, a bird that steers by three rules toward, away from, and along its neighbours.

    Every step a boid sees the boids within `radius`. `FlockingInteraction` steers it by cohesion toward their center,
    by separation away from those closer than `separation_radius`, and by alignment toward their mean velocity, with
    its speed between `min_speed` and `max_speed`, and `FlightInteraction` moves it on a torus of side `width` by
    `height`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        vx:
            The initial horizontal velocity.

        vy:
            The initial vertical velocity.

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

        width:
            The width of the torus.

        height:
            The height of the torus.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.collective_behaviour import BoidElement
        >>> GroupElement('flock', element=BoidElement, size=10, location=(5.0, 5.0)).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'vx'}),
        *,
        vx: Annotated[float, StateSpec(role='velocity')] = 1.0,
        vy: Annotated[float, StateSpec(role='velocity')] = 0.0,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        separation_radius: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        cohesion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        separation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        alignment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        min_speed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        max_speed: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
        height: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 30.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.separation_radius = separation_radius
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.width = width
        self.height = height

    @classmethod
    def _get_view(cls) -> type[BoidView]:
        """Return the view of a boid, `BoidView`."""
        return BoidView
