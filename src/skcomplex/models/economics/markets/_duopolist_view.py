"""Define the view of a Cournot duopolist, as the market rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class DuopolistView(AgentView):
    """Duopolist as the market rule sees it: its `output` and its marginal `cost`."""

    output: float
    cost: float
