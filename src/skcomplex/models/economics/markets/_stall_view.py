"""Define the view of a stall of the produce market, as the shoppers see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class StallView(AgentView):
    """Stall as the shoppers see it: its index `stall`, its `stock` of produce as a bit mask, and its `markup`."""

    stall: float
    stock: float
    markup: float
