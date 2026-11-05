"""View of an agent of the naming game, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class NamerView(AgentView):
    """View of an agent of the naming game, with its states and parameters as attributes."""

    words: Array
