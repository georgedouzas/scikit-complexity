"""View of a robot of the swarmtaxis algorithm, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class SwarmRobotView(AgentView):
    """View of a robot of the swarmtaxis algorithm, with its states and parameters as attributes."""

    heading: float
    connections: float
    illuminated: float
    speed: float
    radio_range: float
    alpha: float
    avoid_illuminated: float
    avoid_shadowed: float
    shadow_angle: float
    beacon_x: float
    beacon_y: float
    arena: float
