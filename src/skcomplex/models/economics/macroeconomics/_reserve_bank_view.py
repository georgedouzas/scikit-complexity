"""Define the view of a bank of the fractional-reserve banking model, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class ReserveBankView(AgentView):
    """Bank as the rules see it: its `index`, `reserves`, `deposits`, and `loans`."""

    index: float
    reserves: float
    deposits: float
    loans: float
