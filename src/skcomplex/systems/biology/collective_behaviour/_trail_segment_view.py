"""View of a trail segment as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class TrailSegmentView(AgentView):
    """View of a trail segment, with its states as attributes.

    `TrailSegmentElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    tau: float
    passable: float
