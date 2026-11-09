"""Markets systems: Gooding's double auction.

Its buyers and sellers adapt their bids and asks after every meeting.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.policy import PolicyRuleInteraction


class AdaptiveTraderView(AgentView):
    """Trader as the auction sees it.

    It holds its value or cost, its quote, and the units it can still trade.
    """

    value: float
    quote: float
    units: float


class AdaptiveTraderElement(Element):
    """Buyer or seller of a double auction.

    It holds its value or cost, its current bid or ask, and its units left.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        value:
            The value of a unit to a buyer, or its cost to a seller.

        quote:
            The current bid of a buyer or ask of a seller.

        units:
            The units a buyer still wants or a seller still holds.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import AdaptiveTraderElement
        >>> GroupElement('buyers', element=AdaptiveTraderElement, size=3, values={'value': 120.0}).size
        3
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'quote'}),
        *,
        value: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        quote: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        units: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.value = value
        self.quote = quote
        self.units = units

    @classmethod
    def _get_view(cls) -> type[AdaptiveTraderView]:
        """Return the view of the agents of the element."""
        return AdaptiveTraderView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the auction moves every trader.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class AdaptiveAuctionInteraction(PolicyRuleInteraction[SystemView]):
    """Double auction in which every buyer meets a random seller every iteration.

    Both adapt their quotes. Every iteration the buyers that still want units, in random order, meet a random seller
    that still holds units. When the bid reaches the ask, one unit trades at the ask, and afterwards the buyer lowers
    its bid and the seller raises its ask, to push the price their way. When it does not, the buyer raises its bid and
    the seller lowers its ask, to create trade, each change a uniform draw between 0 and `step`. Bids never exceed the
    values and asks never fall below the costs. With `unlimited` units, traders are resupplied and keep consuming, so
    trade goes on forever.

    Args:
        label:
            The label of the auction.

        rule:
            Unused, since the auction overrides `act`.

        phase:
            The phase the auction acts in, or `None` for the default phase.

        buyers:
            The path of the group of buyers.

        sellers:
            The path of the group of sellers.

        step:
            The change of a quote after a meeting.

        unlimited:
            Whether the units of the traders never run out.

    Examples:
        >>> from skcomplex.systems.economics.markets import AdaptiveAuctionInteraction
        >>> AdaptiveAuctionInteraction('auction', buyers='buyers', sellers='sellers').reports
        ('price', 'volume', 'mean_bid', 'mean_ask')
    """

    draws = True

    reports: ClassVar[tuple[str, ...]] = ('price', 'volume', 'mean_bid', 'mean_ask')

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        buyers: Annotated[str, RoleSpec()],
        sellers: Annotated[str, RoleSpec()],
        step: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        unlimited: bool = False,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.buyers = buyers
        self.sellers = sellers
        self.step = step
        self.unlimited = unlimited

    def act(self, system: SystemView) -> None:
        """Run the meetings of one iteration.

        Args:
            system:
                The system view of the auction.
        """
        rng, step = system.rng, system.parameter('step')
        buyers = sorted(system.agents('buyers', AdaptiveTraderView), key=lambda trader: trader.id)
        sellers = sorted(system.agents('sellers', AdaptiveTraderView), key=lambda trader: trader.id)
        prices = []
        for buyer in (buyers[index] for index in rng.permutation(len(buyers))):
            stocked = [seller for seller in sellers if seller.units > 0]
            if buyer.units <= 0 or not stocked:
                continue
            seller = stocked[int(rng.integers(len(stocked)))]
            bid_step, ask_step = step * rng.random(), step * rng.random()
            if buyer.quote >= seller.quote:
                prices.append(seller.quote)
                if not self.unlimited:
                    buyer.units, seller.units = buyer.units - 1, seller.units - 1
                buyer.quote = max(buyer.quote - bid_step, 0.0)
                seller.quote += ask_step
            else:
                buyer.quote = min(buyer.quote + bid_step, buyer.value)
                seller.quote = max(seller.quote - ask_step, seller.value)
        last = system.history('price', 1)
        price = prices[-1] if prices else (float(last[-1]) if len(last) else float('nan'))
        system.report('price', price)
        system.report('volume', float(len(prices)))
        system.report('mean_bid', float(np.mean([buyer.quote for buyer in buyers])))
        system.report('mean_ask', float(np.mean([seller.quote for seller in sellers])))


class GoodingDoubleAuctionSystem(BaseSystem):
    """Gooding's double auction.

    Its buyers and sellers move their quotes towards trade and then apart. Buyers value a unit near `value` and sellers
    can supply it near `cost`, each differing by up to `spread` of it. As in the experiments the model replicates,
    buyers start bidding near the sellers' cost and sellers start asking near the buyers' value, and both move their
    quotes towards one another until they trade, after which each tries to push the price its way (Gooding 2019, Section
    11.3 and the appendix). In the short run, with sellers of a few units and buyers that are satiated after a few, the
    price settles near the competitive price halfway between value and cost, 70 at the defaults. With unlimited supply
    and demand, the long-run price wanders between the cost and the value and the equilibrium disappears. The source
    describes the rules in prose, so the meetings and the fixed steps of the quotes are a reconstruction of them.

    The system has the groups `buyers` and `sellers` and the auction `auction`, whose reporters are the last traded
    price `auction.price`, the units traded `auction.volume`, and the mean quotes `auction.mean_bid` and
    `auction.mean_ask`.

    Args:
        n_buyers:
            The number of buyers.

        n_sellers:
            The number of sellers.

        value:
            The mean value of a unit to the buyers.

        cost:
            The mean cost of a unit to the sellers.

        spread:
            The largest relative difference of a trader's value or cost from the mean.

        units:
            The units every seller holds and every buyer wants.

        step:
            The change of a quote after a meeting.

        unlimited:
            Whether the units of the traders never run out.

        random_state:
            The seed or generator of the values and costs.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import GoodingDoubleAuctionSystem
        >>> [group.prepare().size_ for group in GoodingDoubleAuctionSystem(4, 3).prepare().system_.elements]
        [4, 3]
    """

    field = 'economics/markets'
    topics = ('double auction', 'agents', 'price discovery', 'equilibrium', 'adaptive quotes')
    references = ('Gooding 2019',)
    until = 300

    def __init__(
        self,
        n_buyers: int = 100,
        n_sellers: int = 100,
        *,
        value: float = 120.0,
        cost: float = 20.0,
        spread: float = 0.1,
        units: float = 5.0,
        step: float = 1.0,
        unlimited: bool = False,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_buyers = n_buyers
        self.n_sellers = n_sellers
        self.value = value
        self.cost = cost
        self.spread = spread
        self.units = units
        self.step = step
        self.unlimited = unlimited
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        values = self.value * (1 + self.spread * rng.uniform(-1, 1, self.n_buyers))
        costs = self.cost * (1 + self.spread * rng.uniform(-1, 1, self.n_sellers))
        buyers = GroupElement(
            'buyers',
            element=AdaptiveTraderElement,
            size=self.n_buyers,
            values={
                'value': values,
                'quote': self.cost * (1 + self.spread * rng.uniform(-1, 1, self.n_buyers)),
                'units': self.units,
            },
        )
        sellers = GroupElement(
            'sellers',
            element=AdaptiveTraderElement,
            size=self.n_sellers,
            values={
                'value': costs,
                'quote': self.value * (1 + self.spread * rng.uniform(-1, 1, self.n_sellers)),
                'units': self.units,
            },
        )
        auction = AdaptiveAuctionInteraction(
            'auction',
            buyers='buyers',
            sellers='sellers',
            step=self.step,
            unlimited=self.unlimited,
        )
        return CompositeSystem(self.label, elements=[buyers, sellers], interactions=[auction])
