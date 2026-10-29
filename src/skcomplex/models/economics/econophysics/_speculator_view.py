"""Define the view of a speculator of the minority game, as the rules of the game see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class SpeculatorView(AgentView):
    """Speculator as the minority game sees it, with its strategies, action, and wealth.

    It reads its `first` and `second` strategies, their scores `first_score` and `second_score`, its last `action`,
    and its `wealth`.
    """

    first: float
    second: float
    first_score: float
    second_score: float
    action: float
    wealth: float
