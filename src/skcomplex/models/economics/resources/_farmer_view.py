"""Define the view of a farmer of the harvesting model, as its catch rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class FarmerView(AgentView):
    """Farmer as its catch rule sees it: its `wealth`."""

    wealth: float
