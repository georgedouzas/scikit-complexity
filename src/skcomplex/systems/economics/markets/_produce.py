"""Markets systems: shoppers who buy their lists of produce from market stalls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ObservableSpec, StateSpec, VisualSpec
from ....core.types import Array


class ShopperView(AgentView):
    """Shopper as its rule sees it.

    It holds its `wanted` items, its `visited` stalls, its `missed` items, and its `spent`.
    """

    wanted: float
    visited: float
    missed: float
    spent: float


class StallView(AgentView):
    """Stall as the shoppers see it.

    It holds its index `stall`, its `stock` of produce as a bit mask, and its `markup`.
    """

    stall: float
    stock: float
    markup: float


WHOLESALE_PRICES = (0.6, 0.5, 0.7, 0.9, 1.2, 1.5, 0.8, 0.4, 0.5, 0.3, 0.6, 1.0)


class ShopperElement(Element):
    """Shopper who visits one stall a step and buys what it wants there.

    It only visits stalls it has not visited yet. A shopper gives up the items that it has not found once it has visited
    every stall.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        wanted:
            The bit mask of the items on the shopping list.

        visited:
            The bit mask of the stalls already visited.

        missed:
            The bit mask of the items given up.

        spent:
            The money spent.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import ShopperElement
        >>> GroupElement('shoppers', element=ShopperElement, size=5, values={'wanted': 3}).size
        5
    """

    law_classes = (RuleLaw,)

    stalls: ClassVar[str] = 'stalls'

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        wanted: Annotated[float, StateSpec(domain=NN)] = 0,
        visited: Annotated[float, StateSpec(domain=NN)] = 0,
        missed: Annotated[float, StateSpec(domain=NN)] = 0,
        spent: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wanted = wanted
        self.visited = visited
        self.missed = missed
        self.spent = spent

    @classmethod
    def _get_view(cls) -> type[ShopperView]:
        """Return the view of a shopper, with its list, visits, misses, and spending."""
        return ShopperView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the shopping rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.shop)]

    def shop(self, agent: ShopperView, system: SystemView) -> None:
        """Visit a random stall not visited yet and buy every wanted item that it sells.

        Args:
            agent:
                The shopper.

            system:
                The system.
        """
        wanted, visited = int(agent.wanted), int(agent.visited)
        if not wanted:
            return
        unvisited = [stall for stall in system.agents(self.stalls, StallView) if not visited >> int(stall.stall) & 1]
        if not unvisited:
            agent.missed, agent.wanted = wanted, 0
            return
        stall = unvisited[int(system.rng.integers(len(unvisited)))]
        bought = wanted & int(stall.stock)
        agent.visited = visited | 1 << int(stall.stall)
        agent.spent += (1 + stall.markup) * sum(WHOLESALE_PRICES[item] for item in ShopperElement._items_of(bought))
        agent.wanted = wanted & ~bought

    @staticmethod
    def _items_of(mask: int) -> list[int]:
        """Return the indices of the items in a bit mask."""
        return [item for item in range(len(WHOLESALE_PRICES)) if mask >> item & 1]


class StallElement(Element):
    """Market stall that sells some kinds of produce.

    It sells at the wholesale price plus its markup.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        stall:
            The index of the stall.

        stock:
            The bit mask of the kinds of produce that the stall sells.

        markup:
            The relative markup on the wholesale prices.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import StallElement
        >>> GroupElement('stalls', element=StallElement, size=3, values={'stall': [0, 1, 2]}).size
        3
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        stall: Annotated[float, StateSpec(domain=NN)] = 0,
        stock: Annotated[float, StateSpec(domain=NN)] = 0,
        markup: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stall = stall
        self.stock = stock
        self.markup = markup

    @classmethod
    def _get_view(cls) -> type[StallView]:
        """Return the view of a stall, with its index, stock, and markup."""
        return StallView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since stalls only wait for shoppers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


MAX_LIST_LENGTH = 8


MAX_MARKUP = 30


def _items_left(wanted: object) -> float:
    """Return the mean number of items left on the shoppers' lists, given as bit masks."""
    return float(np.mean([len(ShopperElement._items_of(int(mask))) for mask in np.asarray(wanted, dtype=float)]))


class ProduceMarketSystem(BaseSystem):
    """Hamill and Gilbert's produce market.

    Shoppers with lists of produce visit stalls at random. There are twelve kinds of fruit and vegetables. Every stall
    sells a random selection of them at its own markup of 1 to 30 percent on the wholesale prices, and every shopper
    wants between one and eight of them. The system has the groups `stalls` and `shoppers`, the mean number of items
    left on the lists `items_left`, and the mean money spent `spent`.

    Args:
        n_shoppers:
            The number of shoppers.

        n_stalls:
            The number of stalls.

        n_items_stocked:
            The number of kinds of produce that every stall sells.

        random_state:
            The seed or generator of the stocks, the markups, and the shopping lists.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import ProduceMarketSystem
        >>> [group.size for group in ProduceMarketSystem(10, 3).prepare().system_.elements]
        [3, 10]
    """

    field = 'economics/markets'
    topics = ('market', 'agents', 'heterogeneity')
    references = ('Hamill and Gilbert 2016',)
    until = 10
    observables = (
        ObservableSpec('items_left', group='shoppers', variable='wanted', reduce=_items_left),
        ObservableSpec('spent', group='shoppers', variable='spent'),
    )

    def __init__(
        self,
        n_shoppers: int = 50,
        n_stalls: int = 9,
        *,
        n_items_stocked: int = 6,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_shoppers = n_shoppers
        self.n_stalls = n_stalls
        self.n_items_stocked = n_items_stocked
        self.random_state = random_state

    @staticmethod
    def _random_masks(sizes: npt.NDArray[np.int64], rng: np.random.Generator) -> Array:
        """Return one bit mask per size.

        Each mask has that many distinct items chosen at random.
        """
        n_items = len(WHOLESALE_PRICES)
        masks = [sum(1 << int(item) for item in rng.choice(n_items, size, replace=False)) for size in sizes]
        return np.array(masks, dtype=float)

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        stocks = self._random_masks(np.full(self.n_stalls, self.n_items_stocked), rng)
        markups = rng.integers(1, MAX_MARKUP + 1, self.n_stalls) / 100
        lists = self._random_masks(rng.integers(1, MAX_LIST_LENGTH + 1, self.n_shoppers), rng)
        stalls = GroupElement(
            'stalls',
            element=StallElement,
            size=self.n_stalls,
            values={'stall': np.arange(self.n_stalls, dtype=float), 'stock': stocks, 'markup': markups},
        )
        shoppers = GroupElement('shoppers', element=ShopperElement, size=self.n_shoppers, values={'wanted': lists})
        return System(
            self.label,
            elements=[stalls, shoppers],
        )
