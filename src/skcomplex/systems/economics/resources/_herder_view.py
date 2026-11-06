"""Define the view of a herder of the commons, as its rule, the grazing, and the regulator see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class HerderView(AgentView):
    """Herder as the rules see it, with its income, herd, and quota.

    It reads its `income` and `herd` of the step, the `upkeep` and the `newborn` energy of a cow, and its `quota`.
    """

    income: float
    herd: float
    upkeep: float
    newborn: float
    quota: float
