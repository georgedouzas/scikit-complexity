"""View of a typed word of the random typewriting system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class TypedWordView(AgentView):
    """View of a typed word of the random typewriting system, with its states as attributes."""

    count: float
    length: float
    code: float
