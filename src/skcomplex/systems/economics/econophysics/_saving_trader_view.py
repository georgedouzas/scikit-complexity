"""Define the view of a trader of the kinetic wealth exchange systems, as their trading rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class SavingTraderView(AgentView):
    """Trader as the trading rules see it: its `wealth`, negative down to the debt limit."""

    wealth: float
