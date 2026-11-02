"""Axelrod's model of the dissemination of culture on a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from ....core.elements import GroupElement
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....specs import ObservableSpec
from ....types import AnyArray
from ..._catalog import register_model
from ._cultural_influence import CulturalInfluenceInteraction
from ._village import VillageElement


def largest_region(locations: npt.ArrayLike, cultures: npt.ArrayLike, width: int, height: int) -> float:
    """Return the fraction of the cells in the largest connected region of one culture.

    Args:
        locations:
            The cell of every agent, one `(x, y)` row per agent.

        cultures:
            The culture code of every agent.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

    Returns:
        fraction:
            The size of the largest region of adjacent cells with the same culture over the number of cells.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import largest_region
        >>> largest_region([(0, 0), (1, 0), (0, 1), (1, 1)], [1, 1, 2, 3], 2, 2)
        0.5
    """
    cells = np.asarray(locations, dtype=np.int64).reshape(-1, 2)
    grid = np.full((width, height), -1.0)
    grid[cells[:, 0], cells[:, 1]] = np.asarray(cultures, dtype=float)
    largest = 0
    for culture in np.unique(grid[grid >= 0]):
        labels, count = ndimage.label(grid == culture)
        if count:
            largest = max(largest, int(np.bincount(labels.ravel())[1:].max()))
    return largest / (width * height)


def _count_cultures(cultures: AnyArray) -> int:
    """Return the number of distinct cultures."""
    return len(np.unique(cultures))


def culture_observables(width: int, height: int) -> list[ObservableSpec]:
    """Return the share of the largest cultural region and the number of cultures of a full grid of villages.

    Args:
        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

    Returns:
        observables:
            The observables `largest_region` and `cultures`, each reducing the cultures of the group `villages`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import culture_observables
        >>> [item.label for item in culture_observables(5, 4)]
        ['largest_region', 'cultures']
    """
    locations = [(x, y) for x in range(width) for y in range(height)]
    region = partial(largest_region, locations, width=width, height=height)
    return [
        ObservableSpec('largest_region', group='villages', variable='culture', reduce=region),
        ObservableSpec('cultures', group='villages', variable='culture', reduce=_count_cultures),
    ]


def _get_culture_observables(system: System) -> list[ObservableSpec]:
    """Return the culture observables of the grid of a model."""
    if not isinstance(system.space, LatticeSpace):
        return []
    width, height = (int(size) for size in np.atleast_1d(system.space.shape))
    return culture_observables(width, height)


@register_model(
    field='social/opinion_dynamics',
    tags=('culture', 'agents', 'grid', 'social influence', 'phase transition'),
    references=('Axelrod 1997', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 200},
    observables=_get_culture_observables,
)
def axelrod(
    width: int = 20,
    height: int = 20,
    *,
    features: int = 5,
    traits: int = 10,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Axelrod's culture dissemination model on a full grid with von Neumann neighbourhoods.

    Every village starts with random traits. For few traits per feature the villages converge to one culture, and
    for many traits they freeze into many small cultural regions, an order-disorder transition at a critical `q`.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        features:
            The number of cultural features `F`.

        traits:
            The number of traits `q` of every feature.

        random_state:
            The seed or generator of the initial cultures.

    Returns:
        system:
            The model, with the group `villages` and the influence `influence`. `culture_observables` gives the
            share of the largest cultural region and the number of cultures.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import axelrod
        >>> axelrod(5, 4).elements[0].check().prepare().size_
        20
    """
    rng = np.random.default_rng(random_state)
    size = width * height
    digits = rng.integers(traits, size=(size, features))
    cultures = (digits * traits ** np.arange(features)).sum(axis=1).astype(float)
    locations = [(x, y) for x in range(width) for y in range(height)]
    villages = GroupElement(
        'villages',
        element=VillageElement,
        size=size,
        values={'culture': cultures, 'features': features, 'traits': traits},
        location=locations,
    )
    return System(
        elements=[villages],
        interactions=[CulturalInfluenceInteraction('influence', target='villages')],
        space=LatticeSpace((width, height), boundary='fixed', neighborhood='von_neumann', capacity=1),
    )
