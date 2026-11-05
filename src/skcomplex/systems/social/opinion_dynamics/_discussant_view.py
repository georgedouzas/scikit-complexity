"""View of an agent of the Deffuant bounded confidence system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class DiscussantView(AgentView):
    """View of an agent of the Deffuant bounded confidence system, with its states and parameters as attributes."""

    opinion: float
    confidence: float
    convergence: float
