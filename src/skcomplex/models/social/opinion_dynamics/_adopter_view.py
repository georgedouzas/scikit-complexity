"""View of an agent of the threshold adoption model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class AdopterView(AgentView):
    """View of an agent of the threshold adoption model, with its states and parameters as attributes."""

    adopted: float
    threshold: float
    reach: float
    shifting: float
