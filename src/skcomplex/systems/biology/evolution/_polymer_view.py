"""View of a polymer as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class PolymerView(AgentView):
    """View of a polymer, with its states as attributes.

    `PolymerElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    length: float
    born: float
