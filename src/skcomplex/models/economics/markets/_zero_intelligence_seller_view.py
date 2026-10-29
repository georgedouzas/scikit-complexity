"""Define the view of a zero-intelligence seller, as the rules of the double auction see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ZeroIntelligenceSellerView(AgentView):
    """Zero-intelligence seller as the rules see it: its cost `value`, `ask`, `supply`, and `profit`."""

    value: float
    ask: float
    supply: float
    profit: float
