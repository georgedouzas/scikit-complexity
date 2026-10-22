"""Ecology models: organisms that divide, die, and hop between patches of space, and gather in clusters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Jump, Law, Symbols
from ....core.runtime import register_model
from ....core.space import LatticeSpace
from ....core.system import Observable, System


def _neighbour_pairs(shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """Return the pairs of flat indices of every patch and each of its nearest neighbours on a periodic lattice."""
    indices = np.arange(int(np.prod(shape))).reshape(shape)
    pairs: list[tuple[int, int]] = []
    for axis in range(len(shape)):
        for step in (1, -1):
            neighbours = np.roll(indices, -step, axis=axis)
            pairs.extend(zip(indices.ravel().tolist(), neighbours.ravel().tolist(), strict=True))
    return pairs


class Patch(Element):
    """Patch of space whose organisms divide and die at the same rate, so their number makes a symmetric random walk.

    Every organism divides at the rate `alpha` and dies at the rate `alpha`, and an empty patch stays empty without
    arrivals from its neighbours.

    Args:
        label:
            The label of the patch or the lattice of patches.

        n:
            The initial number of organisms.

        alpha:
            The rate at which an organism divides, equal to the rate at which it dies.

    Examples:
        >>> from skcomplex.models.biology.ecology import Patch
        >>> Patch('patch', n=5.0).alpha
        1.0
    """

    visual: ClassVar[VisualSpec] = VisualSpec('image', {'low': 0.0})
    n: float = state(default=10.0, domain=Domain.NONNEGATIVE_INTEGER)
    alpha: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the divisions and deaths of the organisms of the patch.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Jump(s.alpha * s.n, {s.n: 1}), Jump(s.alpha * s.n, {s.n: -1})]


class Hopping(Interaction):
    """Hops of single organisms between neighbouring patches, each organism moving to each neighbour at the rate `beta`.

    The patches are the entries of a variadic role, and every pair `(i, j)` of `links` gives a jump that moves one
    organism from the patch of entry `i` to the patch of entry `j` at the rate `beta n_i`.

    Args:
        label:
            The label of the interaction.

        patches:
            The paths of the patches, such as the members `patches[i]` of a lattice.

        links:
            The ordered pairs of entries of `patches` between which organisms hop.

        beta:
            The rate at which an organism hops to one neighbouring patch.

    Examples:
        >>> from skcomplex.models.biology.ecology import Hopping
        >>> Hopping('hopping', patches=['a', 'b'], links=[(0, 1), (1, 0)]).beta
        0.5
    """

    patches: list[str] = role(variadic=True)
    beta: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    links: Sequence[tuple[int, int]] = ()

    def laws(self, s: Symbols) -> list[Law]:
        """Return one hop per ordered pair of neighbouring patches.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        patches = s.patches
        return [Jump(s.beta * patches[i].n, {patches[i].n: -1, patches[j].n: 1}) for i, j in self.links]


@register_model(
    field='biology/ecology',
    tags=('neutral theory', 'clustering', 'jump process', 'lattice', 'birth-death', 'diffusion'),
    references=('Bertin 2021', 'Houchmandzadeh 2002', 'Houchmandzadeh 2009'),
    defaults={'until': 50.0, 'dt': 0.5, 'backend': 'numba'},
    observables=lambda system: [
        Observable('mean', group='patches', variable='n', reduce='mean'),
        Observable('variance', group='patches', variable='n', reduce='var'),
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
        >>> neutral_clustering(8).elements[0].size
        8
    """
    lattice = (shape,) if isinstance(shape, int) else tuple(shape)
    patches = Patch.lattice('patches', LatticeSpace(lattice), n=np.full(lattice, float(n)), alpha=alpha)
    members = [f'patches[{index}]' for index in range(int(np.prod(lattice)))]
    interactions = []
    if beta:
        interactions.append(Hopping('hopping', patches=members, links=_neighbour_pairs(lattice), beta=beta))
    return System(
        components=[patches],
        interactions=interactions,
    )
