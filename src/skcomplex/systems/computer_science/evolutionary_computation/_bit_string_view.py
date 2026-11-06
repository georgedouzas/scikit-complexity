"""View of an individual of a fitness landscape, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class BitStringView(AgentView):
    """View of an individual of a fitness landscape, with its states as attributes."""

    genome: float
    fitness: float
