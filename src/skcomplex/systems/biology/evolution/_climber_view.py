"""View of a climber as its rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class ClimberView(AgentView):
    """View of a climber, with its states and parameters as attributes.

    `ClimberElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    genome: float
    fitness: float
    n: float
    k: float
    seed: float
