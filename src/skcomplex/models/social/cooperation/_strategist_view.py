"""View of a strategist of the iterated prisoner's dilemma, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView
from ....types import Array


class StrategistView(AgentView):
    """View of a strategist of the iterated prisoner's dilemma, with its states as attributes."""

    genome: Array
    fitness: float
