"""Define the view of a zero-intelligence buyer, as the rules of the double auction see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ZeroIntelligenceBuyerView(AgentView):
    """Zero-intelligence buyer as the rules see it, with its value, bid, demand, profit, and price bounds.

    It reads its `value`, `bid`, `demand`, and `profit`, the price `ceiling`, and whether it is `constrained`.
    """

    value: float
    bid: float
    demand: float
    profit: float
    ceiling: float
    constrained: float
