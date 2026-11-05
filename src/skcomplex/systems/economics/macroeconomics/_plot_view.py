"""Define the view of a patch of Metaphoria, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class PlotView(AgentView):
    """Patch as the rules see it: its `food`, `capacity`, `resistance`, and `regrowth`."""

    food: float
    capacity: float
    resistance: float
    regrowth: float
