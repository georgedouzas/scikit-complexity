"""Define the view of a trader of Wilhite's barter, as the barter rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class BarterTraderView(AgentView):
    """Barter trader as the rules see it: its holdings `good1` and `good2`, its rate `mrs`, `links`, and `partners`."""

    good1: float
    good2: float
    mrs: float
    links: float
    partners: float
