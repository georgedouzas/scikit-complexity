"""Define the view of the till of the car wash, as the operation sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class TillView(AgentView):
    """Till as the operation sees it: its `revenue` and its counts of `washes`, `valets`, and cars `turned_away`."""

    revenue: float
    washes: float
    valets: float
    turned_away: float
