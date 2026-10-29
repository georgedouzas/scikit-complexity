"""Define the view of a shop of the perishable goods duopoly, as the trade sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ShopView(AgentView):
    """Shop as the trade sees it: its `share` of customers and the mean `freshness` of its stock."""

    share: float
    freshness: float
