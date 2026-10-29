"""View of an agent of the Sznajd model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class PersuaderView(AgentView):
    """View of an agent of the Sznajd model, with its states and parameters as attributes."""

    opinion: float
