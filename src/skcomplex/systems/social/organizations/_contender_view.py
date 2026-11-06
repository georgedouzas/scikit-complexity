"""View of an agent of the Bonabeau hierarchy system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ContenderView(AgentView):
    """View of an agent of the Bonabeau hierarchy system, with its states and parameters as attributes."""

    strength: float
    eta: float
    relaxation: float
