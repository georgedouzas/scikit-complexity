"""View of an agent of the cognitive consistency system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class ThinkerView(AgentView):
    """View of an agent of the cognitive consistency system, with its states and parameters as attributes."""

    valences: Array
    attitude: float
    decay: float
