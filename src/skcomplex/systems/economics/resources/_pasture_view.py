"""Define the view of a pasture of the commons, as the grazing and the regulator see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class PastureView(AgentView):
    """Pasture as the rules see it: its `grass`, its `capacity`, and the `regrowth` and `seed` of its grass."""

    grass: float
    capacity: float
    regrowth: float
    seed: float
