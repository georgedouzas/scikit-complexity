"""View of a word of the Simon system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class GrowingWordView(AgentView):
    """View of a word of the Simon system, with its states and parameters as attributes."""

    count: float
    innovation: float
