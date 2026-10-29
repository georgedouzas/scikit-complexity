"""View of an ageing walker, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class AgeingWalkerView(AgentView):
    """View of an ageing walker, with its states and parameters as attributes."""

    x: float
    ups: float
    p: float
    nu: float
