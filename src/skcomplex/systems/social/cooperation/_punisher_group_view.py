"""View of a group of the two-level birth-death process, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class PunisherGroupView(AgentView):
    """View of a group of the two-level birth-death process, with its states as attributes."""

    punishers: float
