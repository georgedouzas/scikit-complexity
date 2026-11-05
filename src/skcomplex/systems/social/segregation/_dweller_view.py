"""View of a dweller of the stochastic Schelling system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class DwellerView(AgentView):
    """View of a dweller of the stochastic Schelling system, with its states and parameters as attributes."""

    block: float
    beta: float
    altruism: float
    slope: float
    intercept: float
