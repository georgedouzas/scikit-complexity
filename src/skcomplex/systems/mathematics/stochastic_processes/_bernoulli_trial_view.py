"""View of a Bernoulli trial, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class BernoulliTrialView(AgentView):
    """View of a Bernoulli trial, with its states and parameters as attributes."""

    outcome: float
    successes: float
    p: float
