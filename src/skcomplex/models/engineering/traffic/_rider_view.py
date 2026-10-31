"""View of a rider of the ridesharing model, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class RiderView(AgentView):
    """View of a rider of the ridesharing model, with its states as attributes."""

    destination_x: float
    destination_y: float
    waited: float
    seats: float
    money: float
