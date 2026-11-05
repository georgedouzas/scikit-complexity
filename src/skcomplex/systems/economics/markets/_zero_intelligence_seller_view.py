"""Define the view of a zero-intelligence seller, as the rules of the double auction see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ZeroIntelligenceSellerView(AgentView):
    """Zero-intelligence seller as the rules see it, with its cost, ask, supply, profit, and price bounds.

    It reads its cost `value`, `ask`, `supply`, and `profit`, the price `ceiling`, and whether it is `constrained`.
    """

    value: float
    ask: float
    supply: float
    profit: float
    ceiling: float
    constrained: float
