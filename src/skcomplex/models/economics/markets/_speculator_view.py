"""Define the view of a speculator of the Brock-Hommes market, as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class SpeculatorView(AgentView):
    """Speculator as the rules see it: its heuristic `predictor` and its intensity of choice `intensity`."""

    predictor: float
    intensity: float
