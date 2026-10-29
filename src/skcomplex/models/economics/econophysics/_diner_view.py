"""Define the view of a diner of the Kolkata restaurant problem, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class DinerView(AgentView):
    """Diner as the rules see it, with its restaurant, the crowd there, and whether it was served.

    It reads its `restaurant`, the `crowd` there, whether it was `served`, and the number of `restaurants`.
    """

    restaurant: float
    crowd: float
    served: float
    restaurants: float
