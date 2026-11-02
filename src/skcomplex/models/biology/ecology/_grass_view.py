"""View of a patch of grass as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class GrassView(AgentView):
    """View of a patch of grass, with its states and parameters as attributes.

    `GrassElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    grown: float
    countdown: float
    regrowth: float
