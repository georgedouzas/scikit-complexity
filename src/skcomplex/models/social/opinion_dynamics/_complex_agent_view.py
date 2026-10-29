"""View of an agent of the standardization model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ComplexAgentView(AgentView):
    """View of an agent of the standardization model, with its states and parameters as attributes."""

    configuration: float
    spin: float
