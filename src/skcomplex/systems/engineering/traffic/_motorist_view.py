"""View of a car of Charbonneau's traffic system, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class MotoristView(AgentView):
    """View of a car of Charbonneau's traffic system, with its states and parameters as attributes."""

    x: float
    v: float
    rank: float
    safe_distance: float
    braking: float
    acceleration: float
    speed_limit: float
