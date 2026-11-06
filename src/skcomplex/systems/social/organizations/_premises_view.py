"""View of a premises of the trafficking system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class PremisesView(AgentView):
    """View of a premises of the trafficking system, with its states and parameters as attributes."""

    brothel: float
    bills: float
    last_bills: float
    efficiency: float
    day: float
