"""View of an agent of the influentials model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView
from ....types import Array


class BelieverView(AgentView):
    """View of an agent of the influentials model, with its states and parameters as attributes."""

    belief: Array
    caring: float
    influence: float
    learning: float
