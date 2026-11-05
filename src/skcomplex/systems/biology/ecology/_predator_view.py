"""View of a predator as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class PredatorView(AgentView):
    """View of a predator, with its state and parameters as attributes.

    `PredatorElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    meals: float
    conversion: float
    death: float
