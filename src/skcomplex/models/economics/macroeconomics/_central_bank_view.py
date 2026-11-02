"""Define the view of the bank of Metaphoria, as its rule sees it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class CentralBankView(AgentView):
    """Bank as its rule sees it: its `money` and its `loans`."""

    money: float
    loans: float
