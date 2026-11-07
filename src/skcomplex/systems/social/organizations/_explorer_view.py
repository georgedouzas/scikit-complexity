"""View of an individual of the mobility systems, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ExplorerView(AgentView):
    """View of an individual of the mobility systems, with its states as attributes."""

    x: float
    y: float
    distinct: float
    level: float
