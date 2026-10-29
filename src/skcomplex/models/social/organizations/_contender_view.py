"""View of an agent of the Bonabeau hierarchy model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ContenderView(AgentView):
    """View of an agent of the Bonabeau hierarchy model, with its states and parameters as attributes."""

    strength: float
    eta: float
    relaxation: float
