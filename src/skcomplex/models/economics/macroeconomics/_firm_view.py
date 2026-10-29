"""Define the view of a firm of the credit network model, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class FirmView(AgentView):
    """Firm as the rules see it: its net worth, demand, loan, rate, partner, gross return, injection, and default."""

    net_worth: float
    demand: float
    loan: float
    rate: float
    partner: float
    gross_return: float
    injected: float
    defaulted: float
