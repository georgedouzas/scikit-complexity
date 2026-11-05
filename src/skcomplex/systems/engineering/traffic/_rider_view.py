"""View of a rider of the ridesharing system, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class RiderView(AgentView):
    """View of a rider of the ridesharing system, with its states as attributes."""

    destination_x: float
    destination_y: float
    waited: float
    seats: float
    money: float
