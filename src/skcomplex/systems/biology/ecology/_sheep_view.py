"""View of a sheep as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class SheepView(AgentView):
    """View of a sheep, with its states and parameters as attributes.

    `SheepElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    energy: float
    gain: float
    reproduction: float
