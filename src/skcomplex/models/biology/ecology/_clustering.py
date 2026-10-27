"""Ecology models: organisms that divide, die, and hop between patches of space, and gather in clusters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, LatticeElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, JumpLaw
from ....core.spaces import LatticeSpace
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.backends import NumbaBackend
from ....engine.runtime import register_model


def _neighbor_pairs(shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """Return the pairs of flat indices of every patch and each of its nearest neighbours on a periodic lattice."""
    indices = np.arange(int(np.prod(shape))).reshape(shape)
    pairs: list[tuple[int, int]] = []
    for axis in range(len(shape)):
        for step in (1, -1):
            neighbours = np.roll(indices, -step, axis=axis)
            pairs.extend(zip(indices.ravel().tolist(), neighbours.ravel().tolist(), strict=True))
    return pairs


class PatchElement(Element):
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
        >>> from skcomplex.models.biology.ecology import PatchElement
        >>> PatchElement('patch', n=5.0).alpha
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('image', {'low': 0.0}),
        *,
        n: Annotated[float, StateSpec(domain=NN)] = 10.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.alpha = alpha

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the divisions and deaths of the organisms of the patch.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [JumpLaw(s.alpha * s.n, {s.n: 1}), JumpLaw(s.alpha * s.n, {s.n: -1})]


class HoppingInteraction(Interaction):
    """Hops of single organisms between neighbouring patches, each organism moving to each neighbour at the rate `beta`.

    The patches are the entries of a variadic role, and every pair `(i, j)` of `links` gives a jump that moves one
    organism from the patch of entry `i` to the patch of entry `j` at the rate `beta n_i`.

    Args:
        label:
            The label of the interaction.

        links:
            The ordered pairs of entries of `patches` between which organisms hop.

        **values:
            The values of the declared variables, by name:

            patches:
                The paths of the patches, such as the members `patches[i]` of a lattice.

            beta:
                The rate at which an organism hops to one neighbouring patch.

    Examples:
        >>> from skcomplex.models.biology.ecology import HoppingInteraction
        >>> HoppingInteraction('hopping', patches=['a', 'b'], links=[(0, 1), (1, 0)]).beta
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        patches: Annotated[list[str], RoleSpec(variadic=True)],
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        links: Sequence[tuple[int, int]] = (),
    ) -> None:
        super().__init__(label)
        self.patches = patches
        self.beta = beta
        self.links = links

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return one hop per ordered pair of neighbouring patches.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        patches = s.patches
        return [JumpLaw(s.beta * patches[i].n, {patches[i].n: -1, patches[j].n: 1}) for i, j in self.links]


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
        >>> neutral_clustering(8).elements[0].size
        8
    """
    lattice = (shape,) if isinstance(shape, int) else tuple(shape)
    patches = LatticeElement(
        'patches',
        element=PatchElement,
        space=LatticeSpace(lattice),
        values={'n': np.full(lattice, float(n)), 'alpha': alpha},
    )
    members = [f'patches[{index}]' for index in range(int(np.prod(lattice)))]
    interactions = []
    if beta:
        interactions.append(HoppingInteraction('hopping', patches=members, links=_neighbor_pairs(lattice), beta=beta))
    return System(
        elements=[patches],
        interactions=interactions,
    )
