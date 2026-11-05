"""Define the view of a lender of the credit network system, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class LenderView(AgentView):
    """Lender as the rules see it: its `equity`, `ask`, `supply`, `lent`, `losses`, `alpha`, and `capital_ratio`."""

    equity: float
    ask: float
    supply: float
    lent: float
    losses: float
    alpha: float
    capital_ratio: float
