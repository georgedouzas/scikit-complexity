"""Define the view of a product of the perishable goods duopoly, as the trade sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ProductView(AgentView):
    """Product as the trade sees it: its `shop` and the time it was `shelved`."""

    shop: float
    shelved: float
