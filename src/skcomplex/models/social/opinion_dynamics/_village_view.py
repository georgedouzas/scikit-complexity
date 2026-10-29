"""View of a village of the Axelrod culture model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class VillageView(AgentView):
    """View of a village of the Axelrod culture model, with its states and parameters as attributes."""

    culture: float
    features: float
    traits: float
