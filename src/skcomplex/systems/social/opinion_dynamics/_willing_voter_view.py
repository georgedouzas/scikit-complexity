"""View of a voter of the majority rule with willingness, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class WillingVoterView(AgentView):
    """View of a voter of the majority rule with willingness, with its states as attributes."""

    opinion: float
    willingness: float
