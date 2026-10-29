"""Define the view of a herder of the commons, as its rule, the grazing, and the regulator see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class HerderView(AgentView):
    """Herder as the rules see it, with its number, income, herd, and quota.

    It reads its `number`, its `income` and `herd` of the step, the `upkeep` and the `newborn` energy of a cow, and
    its `quota`.
    """

    number: float
    income: float
    herd: float
    upkeep: float
    newborn: float
    quota: float
