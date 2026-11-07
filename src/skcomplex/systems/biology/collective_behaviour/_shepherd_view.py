"""View of a shepherd as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ShepherdView(AgentView):
    """View of a shepherd, with its states as attributes.

    `ShepherdElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    driving: float
