"""Define the view of a customer of the fractional-reserve banking model, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class DepositorView(AgentView):
    """Customer as the rules see it: the id of its `bank`, its `deposit`, its `loan`, and its shares of payments.

    The parameters are the `spending` share of the deposit paid every step and the `repayment` share of the loan.
    """

    bank: float
    deposit: float
    loan: float
    spending: float
    repayment: float
