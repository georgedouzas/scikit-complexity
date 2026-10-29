"""View of a plant as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class PlantView(AgentView):
    """View of a plant, with its states and parameters as attributes.

    `PlantElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    age: float
    maturity: float
