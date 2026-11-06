"""Define the view of a farmer of the harvesting system, as its catch rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class FarmerView(AgentView):
    """Farmer as its catch rule sees it: its `wealth`."""

    wealth: float
