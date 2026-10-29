"""View of a strategy of a genetic algorithm, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView
from ....types import IntArray


class StrategyView(AgentView):
    """View of a strategy of a genetic algorithm, with its states as attributes."""

    actions: IntArray
    score: float
