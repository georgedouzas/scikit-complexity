"""View of an agent of the Galam majority system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class CitizenView(AgentView):
    """View of an agent of the Galam majority system, with its states and parameters as attributes."""

    opinion: float
