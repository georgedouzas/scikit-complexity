"""View of an agent of the social impact system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class InfluenceeView(AgentView):
    """View of an agent of the social impact system, with its states and parameters as attributes."""

    opinion: float
    persuasiveness: float
    supportiveness: float
    exponent: float
    noise: float
