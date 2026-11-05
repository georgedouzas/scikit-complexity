"""View of an agent of the standardization system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ComplexAgentView(AgentView):
    """View of an agent of the standardization system, with its states and parameters as attributes."""

    configuration: float
    spin: float
