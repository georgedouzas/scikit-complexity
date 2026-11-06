"""View of a strategist of the iterated prisoner's dilemma, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class StrategistView(AgentView):
    """View of a strategist of the iterated prisoner's dilemma, with its states as attributes."""

    genome: Array
    fitness: float
