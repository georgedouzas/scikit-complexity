"""Define the view of a bank of the credit network model, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class BankView(AgentView):
    """Bank as the rules see it: its `equity`, `ask`, `supply`, `lent`, and `losses`."""

    equity: float
    ask: float
    supply: float
    lent: float
    losses: float
