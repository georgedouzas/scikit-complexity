"""View of a vessel as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class VesselView(AgentView):
    """View of a vessel, with its parameters as attributes.

    `VesselElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    supply: float
