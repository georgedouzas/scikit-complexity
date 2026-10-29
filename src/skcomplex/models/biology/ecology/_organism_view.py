"""View of an organism as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class OrganismView(AgentView):
    """View of an organism, with its states as attributes.

    `OrganismElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    species: float
