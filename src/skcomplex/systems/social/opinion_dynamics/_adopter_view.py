"""View of an agent of the threshold adoption system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class AdopterView(AgentView):
    """View of an agent of the threshold adoption system, with its states and parameters as attributes."""

    adopted: float
    threshold: float
    reach: float
