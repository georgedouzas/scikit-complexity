"""View of an agent of the influentials system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class BelieverView(AgentView):
    """View of an agent of the influentials system, with its states and parameters as attributes."""

    belief: Array
    caring: float
    influence: float
    learning: float
