"""View of an agent of the Deffuant bounded confidence model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class DiscussantView(AgentView):
    """View of an agent of the Deffuant bounded confidence model, with its states and parameters as attributes."""

    opinion: float
    confidence: float
    convergence: float
