"""Define the view of a player of the minority game, as the rules of the game see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class MinorityPlayerView(AgentView):
    """Player as the minority game sees it, with its strategies, action, and wealth.

    It reads its `first` and `second` strategies, their scores `first_score` and `second_score`, the `strategy` it
    plays, 0 for the first and 1 for the second, its last `action`, and its `wealth`.
    """

    first: float
    second: float
    first_score: float
    second_score: float
    strategy: float
    action: float
    wealth: float
