"""Define the view of a forager of the Sugarscape, as its rules and the census see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ForagerView(AgentView):
    """Forager as the rules see it: its wealth `sugar`, its `vision`, and its `metabolism`."""

    sugar: float
    vision: float
    metabolism: float
