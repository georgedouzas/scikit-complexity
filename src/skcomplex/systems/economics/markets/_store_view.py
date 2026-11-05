"""Define the view of a shop of the shops system, as its rules and the market see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class StoreView(AgentView):
    """Shop as the rules see it: its prices, sales, and accounts, and its cost parameters.

    The states are the `price`, the `ask` of the step, the `supply` left, the `sales`, the `money`, the `profit`, and
    the `direction` of the last change of the price. The parameters are the `capacity`, the `unit_cost`, the
    `fixed_cost`, and the price `adjustment`.
    """

    price: float
    ask: float
    supply: float
    sales: float
    money: float
    profit: float
    direction: float
    capacity: float
    unit_cost: float
    fixed_cost: float
    adjustment: float
