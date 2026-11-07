"""View of an agent of the echo chamber system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class RadicalView(AgentView):
    """View of an agent of the echo chamber system, with its states as attributes."""

    opinion: float
    activity: float
