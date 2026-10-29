"""View of a word of the Simon model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class GrowingWordView(AgentView):
    """View of a word of the Simon model, with its states and parameters as attributes."""

    count: float
    innovation: float
