"""View of a voter of the three-state opinion system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class UndecidedView(AgentView):
    """View of a voter of the three-state opinion system, with its states and parameters as attributes."""

    opinion: float
    sensitivity: float
