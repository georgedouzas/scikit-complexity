"""View of a village of the Axelrod culture system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class VillageView(AgentView):
    """View of a village of the Axelrod culture system, with its states and parameters as attributes."""

    culture: float
    features: float
    traits: float
