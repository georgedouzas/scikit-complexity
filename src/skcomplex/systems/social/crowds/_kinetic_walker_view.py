"""View of a walker of the kinetic crowd system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class KineticWalkerView(AgentView):
    """View of a walker of the kinetic crowd system, with its states as attributes."""

    x: float
    y: float
    speed: float
    direction: float
    target: float
