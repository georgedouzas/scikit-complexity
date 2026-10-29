"""Define the view of a cow of the commons, as the grazing sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class CowView(AgentView):
    """Cow as the grazing sees it: the number of its `owner` and its `energy`."""

    owner: float
    energy: float
