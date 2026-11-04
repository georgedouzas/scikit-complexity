"""View of a resident of the Kauai population model, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView
from ....types import Array


class ResidentView(AgentView):
    """View of a resident of the Kauai population model, with its states and parameters as attributes."""

    sentiment: float
    weights: Array
    home: Array
    census: float
    work: float
    favourite: float
    visits: float
    influence: float
    radius: float
