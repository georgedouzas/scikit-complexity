"""View of a node of the Watts cascade system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ThresholdNodeView(AgentView):
    """View of a node of the Watts cascade system, with its states and parameters as attributes."""

    active: float
    threshold: float
