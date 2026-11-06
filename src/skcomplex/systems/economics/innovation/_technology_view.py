"""Define the view of a technology of the pool, as the invention sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class TechnologyView(AgentView):
    """Technology as the invention sees it: its `arity`, its truth `table`, and its `cost` in NAND gates."""

    arity: float
    table: float
    cost: float
