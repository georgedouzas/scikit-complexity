"""Define the view of a trader of the Brock-Hommes market, as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class HeuristicTraderView(AgentView):
    """Trader as the rules see it: its heuristic `predictor` and the last fitness of every heuristic."""

    predictor: float
    fundamentalist_fitness: float
    trend_fitness: float
