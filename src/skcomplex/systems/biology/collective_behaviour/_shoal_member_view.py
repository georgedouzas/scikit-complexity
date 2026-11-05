"""View of a member of a shoal as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ShoalMemberView(AgentView):
    """View of a member of a shoal, with its states and parameters as attributes.

    `ShoalMemberElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    vx: float
    vy: float
    omega: float
    deviation: float
    preferred: float
    alpha: float
    rho: float
    speed: float
    max_turn: float
    noise: float
    size: float
