"""Markets models: shoppers who buy their lists of produce from market stalls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import AgentView, ModelView
from ....specs import ObservableSpec, StateSpec, VisualSpec
from ....types import Array

WHOLESALE_PRICES = (0.6, 0.5, 0.7, 0.9, 1.2, 1.5, 0.8, 0.4, 0.5, 0.3, 0.6, 1.0)
MAX_LIST_LENGTH = 8
MAX_MARKUP = 30


def _items_of(mask: int) -> list[int]:
    """Return the indices of the items in a bit mask."""
    return [item for item in range(len(WHOLESALE_PRICES)) if mask >> item & 1]


def _random_masks(sizes: npt.NDArray[np.int64], rng: np.random.Generator) -> Array:
    """Return one bit mask per size, each with that many distinct items chosen at random."""
    n_items = len(WHOLESALE_PRICES)
    masks = [sum(1 << int(item) for item in rng.choice(n_items, size, replace=False)) for size in sizes]
    return np.array(masks, dtype=float)


def _items_left(agent: AgentView) -> float:
    """Return the number of items left on a shopper's list."""
    return float(len(_items_of(int(agent.wanted))))


class StallElement(Element):
    """Market stall that sells some kinds of produce at the wholesale price plus its markup.

    Args:
        label:
            The label of the agent or group.

        stall:
            The index of the stall.

        stock:
            The bit mask of the kinds of produce that the stall sells.

        markup:
            The relative markup on the wholesale prices.

    Examples:
        >>> from skcomplex.models.economics.markets import StallElement
        >>> GroupElement('stalls', element=StallElement, size=3, values={'stall': [0, 1, 2]}).size
        3
    """

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


class ShopperElement(Element):
    """Shopper who visits one stall a step that it has not visited yet and buys what it wants there.

    A shopper gives up the items that it has not found once it has visited every stall.

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

    Examples:
        >>> from skcomplex.models.economics.markets import ShopperElement
        >>> GroupElement('shoppers', element=ShopperElement, size=5, values={'wanted': 3}).size
        5
    """

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

    def shop(self, agent: AgentView, model: ModelView) -> None:
        """Visit a random stall not visited yet and buy every wanted item that it sells.

        Args:
            agent:
                The shopper.

            model:
                The model.
        """
        wanted, visited = int(agent.wanted), int(agent.visited)
        if not wanted:
            return
        unvisited = [stall for stall in model.agents(self.stalls) if not visited >> int(stall.stall) & 1]
        if not unvisited:
            agent.missed, agent.wanted = wanted, 0
            return
        stall = unvisited[int(model.rng.integers(len(unvisited)))]
        bought = wanted & int(stall.stock)
        agent.visited = visited | 1 << int(stall.stall)
        agent.spent += (1 + stall.markup) * sum(WHOLESALE_PRICES[item] for item in _items_of(bought))
        agent.wanted = wanted & ~bought


@register_model(
    field='economics/markets',
    tags=('market', 'agents', 'heterogeneity'),
    references=('Hamill and Gilbert 2016',),
    defaults={'until': 10},
    observables=lambda system: [
        ObservableSpec('items_left', _items_left, group='shoppers'),
        ObservableSpec('spent', group='shoppers', variable='spent'),
    ],
)
def produce_market(
    n_shoppers: int = 50,
    n_stalls: int = 9,
    *,
    n_items_stocked: int = 6,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Hamill and Gilbert's produce market, where shoppers with lists of produce visit stalls at random.

    There are twelve kinds of fruit and vegetables. Every stall sells a random selection of them at its own markup of 1
    to 30 percent on the wholesale prices, and every shopper wants between one and eight of them.

    Args:
        n_shoppers:
            The number of shoppers.

        n_stalls:
            The number of stalls.

        n_items_stocked:
            The number of kinds of produce that every stall sells.

        random_state:
            The seed or generator of the stocks, the markups, and the shopping lists.

    Returns:
        system:
            The model, with the groups `stalls` and `shoppers`, the mean number of items left on the lists
            `items_left`, and the mean money spent `spent`.

    Examples:
        >>> from skcomplex.models.economics.markets import produce_market
        >>> [group.size for group in produce_market(10, 3).elements]
        [3, 10]
    """
    rng = np.random.default_rng(random_state)
    stocks = _random_masks(np.full(n_stalls, n_items_stocked), rng)
    markups = rng.integers(1, MAX_MARKUP + 1, n_stalls) / 100
    lists = _random_masks(rng.integers(1, MAX_LIST_LENGTH + 1, n_shoppers), rng)
    stalls = GroupElement(
        'stalls',
        element=StallElement,
        size=n_stalls,
        values={'stall': np.arange(n_stalls, dtype=float), 'stock': stocks, 'markup': markups},
    )
    shoppers = GroupElement('shoppers', element=ShopperElement, size=n_shoppers, values={'wanted': lists})
    return System(
        elements=[stalls, shoppers],
    )
