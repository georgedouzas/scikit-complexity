"""View of a coevolving species as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class CoevolvingSpeciesView(AgentView):
    """View of a coevolving species, with its states as attributes.

    `CoevolvingSpeciesElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    genome: float
    fitness: float
    moves: float
