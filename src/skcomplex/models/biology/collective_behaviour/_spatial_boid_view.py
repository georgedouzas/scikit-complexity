"""View of a boid in three dimensions as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView
from ....types import Array


class SpatialBoidView(AgentView):
    """View of a boid in three dimensions, with its states and parameters as attributes.

    `SpatialBoidElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    velocity: Array
    radius: float
    separation_radius: float
    cohesion: float
    separation: float
    alignment: float
    min_speed: float
    max_speed: float
