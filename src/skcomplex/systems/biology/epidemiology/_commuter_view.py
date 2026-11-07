"""View of a commuter as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class CommuterView(AgentView):
    """View of a commuter, with its states as attributes.

    `CommuterElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    home: float
    work: float
    health: float
