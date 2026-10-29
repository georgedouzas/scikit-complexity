"""Define the view of a zero-intelligence buyer, as the rules of the double auction see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ZeroIntelligenceBuyerView(AgentView):
    """Zero-intelligence buyer as the rules see it: its `value`, `bid`, `demand`, and `profit`."""

    value: float
    bid: float
    demand: float
    profit: float
