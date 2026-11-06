"""View of a restaurant of the obesity system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class RestaurantView(AgentView):
    """View of a restaurant of the obesity system, with its states and parameters as attributes."""

    kind: float
    sold: float
    sold_high: float
