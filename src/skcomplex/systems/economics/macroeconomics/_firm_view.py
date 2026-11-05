"""Define the view of a firm of the credit network system, as the rules see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView


class FirmView(AgentView):
    """Firm as the rules see it: its balance sheet, loan, partner, return, injection, default, and parameters.

    It reads its `net_worth`, `demand`, `loan`, `rate`, `partner`, `gross_return`, `injected`, and `defaulted`, and the
    parameters `leverage`, `mean_return`, `volatility`, and `entry`.
    """

    net_worth: float
    demand: float
    loan: float
    rate: float
    partner: float
    gross_return: float
    injected: float
    defaulted: float
    leverage: float
    mean_return: float
    volatility: float
    entry: float
