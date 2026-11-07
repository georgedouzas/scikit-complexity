"""View of a herded sheep as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class HerdedSheepView(AgentView):
    """View of a herded sheep, with its states as attributes.

    `HerdedSheepElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    hx: float
    hy: float
