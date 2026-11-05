"""Define the view of a patch of the Sugarscape, as its growth rule and the foragers see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class SugarPatchView(AgentView):
    """Patch as the rules see it: its `sugar`, its `capacity`, and its `growback`."""

    sugar: float
    capacity: float
    growback: float
