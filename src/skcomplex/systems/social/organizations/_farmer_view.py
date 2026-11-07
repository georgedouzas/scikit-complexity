"""View of an employer of Ajani's Choice, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class FarmerView(AgentView):
    """View of an employer of Ajani's Choice, with its states as attributes."""

    wage: float
    workforce: float
    employed: float
    area: float
