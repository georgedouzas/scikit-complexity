"""View of an individual of a branching process, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class IndividualView(AgentView):
    """View of an individual of a branching process, with its states and parameters as attributes."""

    generation: float
    m: float
