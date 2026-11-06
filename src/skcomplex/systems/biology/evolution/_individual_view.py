"""View of an individual as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class IndividualView(AgentView):
    """View of an individual, with its states as attributes.

    `IndividualElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    trait: float
    fitness: float
