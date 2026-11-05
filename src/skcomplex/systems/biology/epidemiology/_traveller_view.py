"""View of a traveller as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class TravellerView(AgentView):
    """View of a traveller, with its states and parameters as attributes.

    `TravellerElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    health: float
    tau: float
    m: float
