"""View of a woman of the trafficking system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class TravellerView(AgentView):
    """View of a woman of the trafficking system, with its states and parameters as attributes."""

    victim: float
    trips: float
