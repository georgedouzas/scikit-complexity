"""View of a task of the priority queue system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class TaskView(AgentView):
    """View of a task of the priority queue system, with its states and parameters as attributes."""

    priority: float
    age: float
    wait: float
    greed: float
