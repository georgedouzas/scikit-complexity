"""Markets systems: shoppers who buy their lists of produce from market stalls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

WHOLESALE_PRICES = (0.6, 0.5, 0.7, 0.9, 1.2, 1.5, 0.8, 0.4, 0.5, 0.3, 0.6, 1.0)
MAX_LIST_LENGTH = 8
MAX_MARKUP = 30


class ShopperView(AgentViewProtocol, Protocol):
    """Shopper as its rule sees it.

    It holds its `wanted` items, its `visited` stalls, its `missed` items, and its `spent`.
    """

    wanted: float
    visited: float
    missed: float
    spent: float


class StallView(AgentViewProtocol, Protocol):
    """Stall as the shoppers see it.

    It holds its index `stall`, its `stock` of produce as a bit mask, and its `markup`.
    """

    stall: float
    stock: float
    markup: float


def _items_of(mask: int) -> list[int]:
    """Return the indices of the items in a bit mask."""
    return [item for item in range(len(WHOLESALE_PRICES)) if mask >> item & 1]


class ShopperElement(Element):
    """Shopper with a shopping list, who visits stalls by `ShoppingInteraction`.

    Args:
        label:
            The label of the agent or group.

        wanted:
            The bit mask of the items on the shopping list.

        visited:
            The bit mask of the stalls already visited.

        missed:
            The bit mask of the items given up.

        spent:
            The money spent.

        values:
            Further values by name, or `None` for none.

        size:
            The number of shoppers, or `None` for one.

        space:
            The space of the shoppers, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.markets import ShopperElement
        >>> ShopperElement('shoppers', wanted=3, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        *,
        wanted: Annotated[object, NN] = 0,
        visited: Annotated[object, NN] = 0,
        missed: Annotated[object, NN] = 0,
        spent: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.wanted = wanted
        self.visited = visited
        self.missed = missed
        self.spent = spent


class ShoppingInteraction(Interaction):
    """Shopping of every shopper, who visits one stall a step and buys what it wants there.

    A shopper only visits stalls it has not visited yet, and gives up the items that it has not found once it has
    visited every stall.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of shoppers.

        stalls:
            The path of the group of stalls.

    Examples:
        >>> from skcomplex.systems.economics.markets import ShoppingInteraction
        >>> ShoppingInteraction('shopping', target='shoppers', stalls='stalls').stalls
        'stalls'
    """

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role, stalls: Role) -> None:
        super().__init__(label)
        self.target = target
        self.stalls = stalls

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the shopping rule of every shopper.

        Args:
            context:
                The namespace of the shoppers and the stalls.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self.shop)]

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
        unvisited = [
            stall for stall in cast(list[StallView], system.get_agents('stalls')) if not visited >> int(stall.stall) & 1
        ]
        if not unvisited:
            agent.missed, agent.wanted = wanted, 0
            return
        stall = unvisited[int(system.rng.integers(len(unvisited)))]
        bought = wanted & int(stall.stock)
        agent.visited = visited | 1 << int(stall.stall)
        agent.spent += (1 + stall.markup) * sum(WHOLESALE_PRICES[item] for item in _items_of(bought))
        agent.wanted = wanted & ~bought


class StallElement(Element):
    """Market stall that sells some kinds of produce.

    It sells at the wholesale price plus its markup.

    Args:
        label:
            The label of the agent or group.

        stall:
            The index of the stall.

        stock:
            The bit mask of the kinds of produce that the stall sells.

        markup:
            The relative markup on the wholesale prices.

        values:
            Further values by name, or `None` for none.

        size:
            The number of stalls, or `None` for one.

        space:
            The space of the stalls, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.markets import StallElement
        >>> StallElement('stalls', stall=[0, 1, 2], size=3).size
        3
    """

    def __init__(
        self,
        label: str,
        *,
        stall: Annotated[object, NN] = 0,
        stock: Annotated[object, NN] = 0,
        markup: Annotated[object, RealSet.closed_open(0, oo)] = 0.1,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.stall = stall
        self.stock = stock
        self.markup = markup


def _get_items_left(wanted: object) -> float:
    """Return the mean number of items left on the shoppers' lists, given as bit masks."""
    return float(np.mean([len(_items_of(int(mask))) for mask in np.asarray(wanted, dtype=float)]))


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        Observable('items_left', group='shoppers', variable='wanted', reduce=_get_items_left),
        Observable('spent', group='shoppers', variable='spent'),
    )

    def __init__(
        self,
        n_shoppers: int = 50,
        n_stalls: int = 9,
        *,
        n_items_stocked: int = 6,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_shoppers = n_shoppers
        self.n_stalls = n_stalls
        self.n_items_stocked = n_items_stocked
        self.random_state = random_state

    @staticmethod
    def _get_random_masks(sizes: npt.NDArray[np.int64], rng: np.random.Generator) -> Array:
        """Return one bit mask per size.

        Each mask has that many distinct items chosen at random.
        """
        n_items = len(WHOLESALE_PRICES)
        masks = [sum(1 << int(item) for item in rng.choice(n_items, size, replace=False)) for size in sizes]
        return np.array(masks, dtype=float)

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        stocks = self._get_random_masks(np.full(self.n_stalls, self.n_items_stocked), rng)
        markups = rng.integers(1, MAX_MARKUP + 1, self.n_stalls) / 100
        lists = self._get_random_masks(rng.integers(1, MAX_LIST_LENGTH + 1, self.n_shoppers), rng)
        stalls = StallElement(
            'stalls',
            stall=np.arange(self.n_stalls, dtype=float),
            stock=stocks,
            markup=markups,
            size=self.n_stalls,
        )
        shoppers = ShopperElement('shoppers', wanted=lists, size=self.n_shoppers)
        self.system_ = System(
            self.label,
            elements=[stalls, shoppers],
            interactions=[ShoppingInteraction('shopping', target='shoppers', stalls='stalls')],
        )
