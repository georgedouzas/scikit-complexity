"""Define the view of a product of the perishable goods duopoly, as the purchase sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ProductView(AgentView):
    """Product as the purchase sees it: the time it was `shelved` and the time `tau_a` over which it ages."""

    shelved: float
    tau_a: float
