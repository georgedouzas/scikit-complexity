"""View of a forager or a predator as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ForagerView(AgentView):
    """View of a forager or a predator, with its states and parameters as attributes.

    `ForagerElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    age: float
    hunger: float
    happiness: float
    distress: float
    experience: float
    maturity: float
    reproduction: float
    starvation: float
    memory: float
    communication: float
    emotions: float
