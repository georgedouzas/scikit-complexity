"""View of a woman of the trafficking model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class TravellerView(AgentView):
    """View of a woman of the trafficking model, with its states and parameters as attributes."""

    victim: float
    trips: float
