"""View of a driver on a ring road, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class DriverView(AgentView):
    """View of a driver on a ring road, with its states and parameters as attributes."""

    speed: float
    rank: float
    acceleration: float
    min_acc: float
    max_acc: float
    speed_limit: float
    eps: float
