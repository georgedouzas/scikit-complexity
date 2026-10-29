"""View of a restaurant of the obesity model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class RestaurantView(AgentView):
    """View of a restaurant of the obesity model, with its states and parameters as attributes."""

    kind: float
    sold: float
    sold_high: float
