"""Noise trader of an order-driven market, who follows the expectation of the trader it is linked to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw
from ....core.engine.solvers import AgentView
from ....core.specs import StateSpec, VisualSpec


class GuruTraderView(AgentView):
    """Trader as the market sees it: its holdings, wealth, expectation, link, volatility, and followers."""

    cash: float
    stocks: float
    wealth: float
    expectation: float
    follows: float
    sigma: float
    followers: float


class GuruTraderElement(Element):
    """Noise trader of an order-driven market, who follows the expectation of the trader it is linked to.

    The trader holds cash and stocks, has one outgoing link to the trader whose expected return it adopts, and expects
    returns of its own volatility `sigma`. The market of the system rewires its link and runs its orders.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        cash:
            The cash of the trader, including the cash held by its limit orders.

        stocks:
            The stocks of the trader, including the stocks held by its limit orders.

        wealth:
            The cash plus the stocks at the price of the day.

        expectation:
            The expected return of the day after imitation.

        follows:
            The index of the trader whose expectation the trader adopts.

        sigma:
            The volatility of the expected returns of the trader.

        followers:
            The number of traders that follow the trader, its in-degree.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.finance import GuruTraderElement
        >>> GroupElement('traders', element=GuruTraderElement, size=3).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'followers'}),
        *,
        cash: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        stocks: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        wealth: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2000.0,
        expectation: Annotated[float, StateSpec()] = 0.0,
        follows: Annotated[float, StateSpec(domain=NN)] = 0,
        sigma: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        followers: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.cash = cash
        self.stocks = stocks
        self.wealth = wealth
        self.expectation = expectation
        self.follows = follows
        self.sigma = sigma
        self.followers = followers

    @classmethod
    def _get_view(cls) -> type[GuruTraderView]:
        """Return the view of the agents of the element."""
        return GuruTraderView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the market rewires and trades for every trader.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
