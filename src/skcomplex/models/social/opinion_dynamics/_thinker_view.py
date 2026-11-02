"""View of an agent of the cognitive consistency model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView
from ....types import Array


class ThinkerView(AgentView):
    """View of an agent of the cognitive consistency model, with its states and parameters as attributes."""

    valences: Array
    attitude: float
    decay: float
