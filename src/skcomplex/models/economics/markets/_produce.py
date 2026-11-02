"""Markets models: shoppers who buy their lists of produce from market stalls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import numpy.typing as npt

from ....core.elements import GroupElement
from ....core.system import System
from ....specs import ObservableSpec
from ....types import Array
from ..._catalog import register_model
from ._shopper import WHOLESALE_PRICES, ShopperElement
from ._stall import StallElement

MAX_LIST_LENGTH = 8


MAX_MARKUP = 30


def _random_masks(sizes: npt.NDArray[np.int64], rng: np.random.Generator) -> Array:
    """Return one bit mask per size, each with that many distinct items chosen at random."""
    n_items = len(WHOLESALE_PRICES)
    masks = [sum(1 << int(item) for item in rng.choice(n_items, size, replace=False)) for size in sizes]
    return np.array(masks, dtype=float)


def _items_left(wanted: object) -> float:
    """Return the mean number of items left on the shoppers' lists, given as bit masks."""
    return float(np.mean([len(ShopperElement._items_of(int(mask))) for mask in np.asarray(wanted, dtype=float)]))


@register_model(
    field='economics/markets',
    tags=('market', 'agents', 'heterogeneity'),
    references=('Hamill and Gilbert 2016',),
    defaults={'until': 10},
    observables=lambda system: [
        ObservableSpec('items_left', group='shoppers', variable='wanted', reduce=_items_left),
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
