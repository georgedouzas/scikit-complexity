"""View of a watchmaker of the Hora and Tempus fable, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class WatchmakerView(AgentView):
    """View of a watchmaker of the Hora and Tempus fable, with its states and parameters as attributes."""

    progress: float
    done: float
    watches: float
    task_size: float
    tasks: float
