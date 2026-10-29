"""View of a wolf as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class WolfView(AgentView):
    """View of a wolf, with its states and parameters as attributes.

    `WolfElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    energy: float
    gain: float
    reproduction: float
