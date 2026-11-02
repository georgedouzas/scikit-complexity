"""View of a node of the Watts cascade model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ThresholdNodeView(AgentView):
    """View of a node of the Watts cascade model, with its states and parameters as attributes."""

    active: float
    threshold: float
