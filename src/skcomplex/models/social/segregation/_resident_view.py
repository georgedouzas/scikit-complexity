"""View of a resident of the Schelling model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ResidentView(AgentView):
    """View of a resident of the Schelling model, with its states and parameters as attributes."""

    kind: float
    threshold: float
