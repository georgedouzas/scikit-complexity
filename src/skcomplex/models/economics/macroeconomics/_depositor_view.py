"""Define the view of a customer of the fractional-reserve banking model, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class DepositorView(AgentView):
    """Customer as the rules see it: the index of its `bank`, its `deposit`, and its `loan`."""

    bank: float
    deposit: float
    loan: float
