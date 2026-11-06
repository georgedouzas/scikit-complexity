"""View of a resident of the Kauai population system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class ResidentView(AgentView):
    """View of a resident of the Kauai population system, with its states and parameters as attributes."""

    sentiment: float
    weights: Array
    home: Array
    census: float
    work: float
    favourite: float
    visits: float
    race: float
    gender: float
    occupation: float
    age: float
    convergence: float
    event_rate: float
    cell_size: float
    partners: float
