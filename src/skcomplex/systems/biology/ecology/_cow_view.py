"""View of a cow as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class CowView(AgentView):
    """View of a cow, with its parameters as attributes.

    `CowElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    bite: float
