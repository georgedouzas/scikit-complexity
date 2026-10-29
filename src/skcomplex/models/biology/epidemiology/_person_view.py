"""View of a person as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class PersonView(AgentView):
    """View of a person, with its states and parameters as attributes.

    `PersonElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    health: float
    clock: float
    d: float
    beta: float
    gamma: float
    mu: float
    alpha: float
    incubation: float
    lockdown_start: float
    lockdown_length: float
    lockdown_d: float
