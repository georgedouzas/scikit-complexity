"""Define the view of a part of a technology, as the redesign sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class PartView(AgentView):
    """Part of a technology as the redesign sees it: its unit `cost`."""

    cost: float
