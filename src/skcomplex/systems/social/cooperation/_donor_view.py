"""View of a player of the game transitions system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class DonorView(AgentView):
    """View of a player of the game transitions system, with its states as attributes."""

    strategy: float
    payoff: float
