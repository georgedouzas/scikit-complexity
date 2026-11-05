"""View of a resident of the Schelling system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ResidentView(AgentView):
    """View of a resident of the Schelling system, with its states and parameters as attributes."""

    kind: float
    threshold: float
