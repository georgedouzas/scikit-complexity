"""Define the view of a diner of the Kolkata restaurant problem, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class DinerView(AgentView):
    """Diner as the rules see it, with its restaurant, the crowd there, and whether it was served.

    It reads its `restaurant`, the `crowd` there, and whether it was `served`.
    """

    restaurant: float
    crowd: float
    served: float
