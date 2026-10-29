"""View of a ball of a sample-space-reducing process, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ReducingBallView(AgentView):
    """View of a ball of a sample-space-reducing process, with its states and parameters as attributes."""

    level: float
    n_states: float
    lam: float
