"""Define the view of a bank of the DebtRank cascade, as the cascade sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class InterbankView(AgentView):
    """Bank as the cascade sees it: its distress `h`, its `status`, and its economic `value`."""

    h: float
    status: float
    value: float
