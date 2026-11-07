"""Define the view of a firm of the Mark-0 economy, as the economy sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class Mark0FirmView(AgentView):
    """Firm as the economy sees it: its workforce, demand, price, wage, cash, profit, and whether it is active."""

    workforce: float
    demand: float
    price: float
    wage: float
    cash: float
    profit: float
    active: float
