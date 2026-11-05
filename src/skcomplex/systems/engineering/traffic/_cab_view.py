"""View of a cab of the ridesharing system, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class CabView(AgentView):
    """View of a cab of the ridesharing system, with its states and parameters as attributes."""

    task: float
    target_x: float
    target_y: float
    destination_x: float
    destination_y: float
    rider: float
    seats: float
    money: float
    base_fare: float
    block_fare: float
