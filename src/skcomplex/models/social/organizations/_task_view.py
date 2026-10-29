"""View of a task of the priority queue model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class TaskView(AgentView):
    """View of a task of the priority queue model, with its states and parameters as attributes."""

    priority: float
    age: float
    wait: float
    greed: float
