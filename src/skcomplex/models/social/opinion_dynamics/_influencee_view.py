"""View of an agent of the social impact model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class InfluenceeView(AgentView):
    """View of an agent of the social impact model, with its states and parameters as attributes."""

    opinion: float
    persuasiveness: float
    supportiveness: float
    exponent: float
    noise: float
