"""View of an agent of the voter model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class VoterView(AgentView):
    """View of an agent of the voter model, with its states and parameters as attributes."""

    opinion: float
