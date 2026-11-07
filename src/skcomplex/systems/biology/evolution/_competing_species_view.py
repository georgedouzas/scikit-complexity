"""View of a competing species as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class CompetingSpeciesView(AgentView):
    """View of a competing species, with its states as attributes.

    `CompetingSpeciesElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    x: float
    links: Array
