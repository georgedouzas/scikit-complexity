"""View of a patch of pasture as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class PastureView(AgentView):
    """View of a patch of pasture, with its states and parameters as attributes.

    `PastureElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    grass: float
    growth: float
