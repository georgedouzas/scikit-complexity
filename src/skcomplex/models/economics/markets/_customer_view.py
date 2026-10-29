"""Define the view of a customer of the perishable goods duopoly, as the trade sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class CustomerView(AgentView):
    """Customer as the trade sees it: its satisfactions `s1` and `s2` with the two shops."""

    s1: float
    s2: float
