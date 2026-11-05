"""View of a boid as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class BoidView(AgentView):
    """View of a boid, with its states and parameters as attributes.

    `BoidElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    vx: float
    vy: float
    radius: float
    separation_radius: float
    cohesion: float
    separation: float
    alignment: float
    min_speed: float
    max_speed: float
    width: float
    height: float
