"""View of an agent of the Sznajd system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class PersuaderView(AgentView):
    """View of an agent of the Sznajd system, with its states and parameters as attributes."""

    opinion: float
