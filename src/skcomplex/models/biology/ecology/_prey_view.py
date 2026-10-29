"""View of a prey as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class PreyView(AgentView):
    """View of a prey, with its parameters as attributes.

    `PreyElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    birth: float
