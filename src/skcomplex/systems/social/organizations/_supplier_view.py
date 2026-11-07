"""View of a firm of the supply chain diffusion system, as its rules read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class SupplierView(AgentView):
    """View of a firm of the supply chain diffusion system, with its states as attributes."""

    adopted: float
    inflow: float
    outflow: float
