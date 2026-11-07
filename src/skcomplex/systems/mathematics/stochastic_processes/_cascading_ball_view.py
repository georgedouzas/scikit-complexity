"""View of a ball of a cascading sample-space-reducing process, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class CascadingBallView(AgentView):
    """View of a ball of a cascading sample-space-reducing process, with its states and parameters as attributes."""

    level: float
    mu: float
