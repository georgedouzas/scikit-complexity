"""Define the view of a shopper of the produce market, as its shopping rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ShopperView(AgentView):
    """Shopper as its rule sees it: its `wanted` items, its `visited` stalls, its `missed` items, and its `spent`."""

    wanted: float
    visited: float
    missed: float
    spent: float
