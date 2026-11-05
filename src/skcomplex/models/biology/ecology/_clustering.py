"""Ecology models: organisms that divide, die, and hop between patches of space, and gather in clusters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.elements import LatticeElement
from ....core.interactions import BaseInteraction, JumpInteraction
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.backends import NumbaBackend
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._habitat_patch import HabitatPatchElement
from ._hopping import HoppingInteraction


def _neighbor_pairs(shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """Return the pairs of flat indices of every patch and each of its nearest neighbours on a periodic lattice."""
    indices = np.arange(int(np.prod(shape))).reshape(shape)
    pairs: list[tuple[int, int]] = []
    for axis in range(len(shape)):
        for step in (1, -1):
            neighbours = np.roll(indices, -step, axis=axis)
            pairs.extend(zip(indices.ravel().tolist(), neighbours.ravel().tolist(), strict=True))
    return pairs


@register_model(
    field='biology/ecology',
    tags=('neutral theory', 'clustering', 'jump process', 'lattice', 'birth-death', 'diffusion'),
    references=('Bertin 2021', 'Houchmandzadeh 2002', 'Houchmandzadeh 2009'),
    defaults={'until': 50.0, 'dt': 0.5, 'backend': NumbaBackend()},
    observables=lambda system: [
        ObservableSpec('mean', group='patches', variable='n', reduce='mean'),
        ObservableSpec('variance', group='patches', variable='n', reduce='var'),
    ],
)
def neutral_clustering(
    shape: int | Sequence[int] = 64,
    *,
    n: float = 10.0,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> System:
    """Return organisms that divide, die, and hop between patches on a periodic lattice, and gather in clusters.

    Births balance deaths, so the mean number of organisms per patch stays `n` while its variance grows. Without hops it
    grows as `2 alpha n t`, so most patches empty while a few fill. Hops at the rate `beta` slow the growth to
    `t^(1/2)` in one dimension and `ln t` in two, so the clusters persist in low dimension although the mean density
    obeys a diffusion equation.

    Args:
        shape:
            The number of patches of a ring, or the shape of a periodic lattice of patches.

        n:
            The initial number of organisms in every patch.

        alpha:
            The rate at which an organism divides, equal to the rate at which it dies.

        beta:
            The rate at which an organism hops to one neighbouring patch.

    Returns:
        system:
            The model, with the numbers `patches.n` of organisms on the lattice `patches`, and their mean `mean` and
            variance `variance` over the patches.

    Examples:
        >>> from skcomplex.models.biology.ecology import neutral_clustering
        >>> neutral_clustering(8).elements[0].prepare().size_
        8
    """
    lattice = (shape,) if isinstance(shape, int) else tuple(shape)
    patches = LatticeElement(
        'patches',
        element=HabitatPatchElement,
        space=LatticeSpace(lattice),
        values={'n': np.full(lattice, float(n)), 'alpha': alpha},
    )
    members = [f'patches[{index}]' for index in range(int(np.prod(lattice)))]
    interactions: list[BaseInteraction] = [
        JumpInteraction('division', target='patches', changes={'n': 1}, rate=lambda s: s.target.alpha * s.target.n),
        JumpInteraction('death', target='patches', changes={'n': -1}, rate=lambda s: s.target.alpha * s.target.n),
    ]
    if beta:
        interactions.append(HoppingInteraction('hopping', patches=members, links=_neighbor_pairs(lattice), beta=beta))
    return System(
        elements=[patches],
        interactions=interactions,
    )
