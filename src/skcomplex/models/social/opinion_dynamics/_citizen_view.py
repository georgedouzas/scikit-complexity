"""View of an agent of the Galam majority model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class CitizenView(AgentView):
    """View of an agent of the Galam majority model, with its states and parameters as attributes."""

    opinion: float
