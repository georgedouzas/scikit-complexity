"""Define the view of a customer of the perishable goods duopoly, as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class CustomerView(AgentView):
    """Customer as its rules see it: its satisfactions `s1` and `s2`, its `shop`, `freshness`, and temperature `T`."""

    s1: float
    s2: float
    shop: float
    freshness: float
    T: float
