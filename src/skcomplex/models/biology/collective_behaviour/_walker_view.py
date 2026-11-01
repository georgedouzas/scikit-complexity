"""View of a walker as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class WalkerView(AgentView):
    """View of a walker, with its states and parameters as attributes.

    `WalkerElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    goal_x: float
    goal_y: float
    arrived: float
    directedness: float
    attraction: float
    intensity: float
    G_max: float
    width: float
    height: float
