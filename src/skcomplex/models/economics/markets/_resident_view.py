"""Define the view of a shopper of the shops model, as its rules and the market see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ResidentView(AgentView):
    """Shopper as the rules see it: its `bid`, `demand`, `money`, and `travel`, and its shopping parameters.

    The parameters are the `distance_cost` of shopping in person and the probability `online` of shopping online.
    """

    bid: float
    demand: float
    money: float
    travel: float
    distance_cost: float
    online: float
