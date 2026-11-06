"""View of an agent of the voter system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class VoterView(AgentView):
    """View of an agent of the voter system, with its states and parameters as attributes."""

    opinion: float
