"""Define the view of a trader of the Boltzmann wealth model, as its exchange rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class TraderView(AgentView):
    """Trader as the exchange rule sees it: its `wealth`, in whole units."""

    wealth: float
