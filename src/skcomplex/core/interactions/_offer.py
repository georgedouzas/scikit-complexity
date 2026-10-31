"""Define the offers of a matching, the limits and quantities of its participants during one clearing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ..types import AgentViewProtocol


class _Offer:
    """Offer of one participant of a matching during one clearing, which the clearing lowers as it matches."""

    def __init__(self, agent: AgentViewProtocol, limit: float, quantity: float, budget: float = math.inf) -> None:
        self.agent = agent
        self.limit = limit
        self.quantity = quantity
        self.budget = budget

    def get_affordable(self, value: float) -> float:
        """Return the quantity the participant can still match at a value within its budget."""
        if value <= 0:
            return self.quantity
        return max(0.0, min(self.quantity, self.budget / value))
