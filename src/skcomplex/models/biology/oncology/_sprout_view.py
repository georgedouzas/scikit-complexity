"""View of a sprout tip as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class SproutView(AgentView):
    """View of a sprout tip, with its parameters as attributes.

    `SproutElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    chemotaxis: float
    branching: float
    activation: float
