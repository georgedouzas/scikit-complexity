"""View of a premises of the trafficking model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class PremisesView(AgentView):
    """View of a premises of the trafficking model, with its states and parameters as attributes."""

    brothel: float
    bills: float
    last_bills: float
    efficiency: float
    day: float
