"""View of an ant as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class AntView(AgentView):
    """View of an ant, with its states and parameters as attributes.

    `AntElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    carrying: float
    arrived: float
    nest_x: float
    nest_y: float
    food_x: float
    food_y: float
    food_radius: float
    attraction: float
    outward: float
    deposit: float
