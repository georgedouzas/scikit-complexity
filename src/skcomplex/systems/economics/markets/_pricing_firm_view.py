"""View of a firm of a discretised pricing game, typed for the rules that read it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class PricingFirmView(AgentView):
    """View of a firm of a pricing game, with its states as attributes."""

    price: float
    scores: Array
