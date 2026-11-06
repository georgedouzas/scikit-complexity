"""Define the view of a firm of Metaphoria, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class BusinessView(AgentView):
    """Firm as the rules see it: its holdings, stock, wage, price, owner, workforce, sales, and policy parameters."""

    money: float
    loans: float
    stock: float
    wage: float
    price: float
    owner: float
    workers: float
    vacancies: float
    revenue: float
    interest: float
    repayment: float
    adjustment: float
    stock_limit: float
