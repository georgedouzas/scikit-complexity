"""Define the view of a stock trader of the order book market, as its rules and the book see it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....engine.solvers import AgentView


class StockTraderView(AgentView):
    """Stock trader as the rules see it, with its strategy, holdings, order, wealth, gain, and fundamental value.

    The order is the limit prices `bid` and `ask` and the quantities `demand` and `supply`.
    """

    strategy: float
    money: float
    shares: float
    bid: float
    ask: float
    demand: float
    supply: float
    wealth: float
    gain: float
    fundamental: float
