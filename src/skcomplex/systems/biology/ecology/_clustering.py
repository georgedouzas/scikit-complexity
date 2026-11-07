"""Ecology systems: organisms that divide, die, and hop between patches of space, and gather in clusters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import CompartmentElement, LatticeElement
from ....core.components.interactions import BaseInteraction, Interaction, JumpInteraction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.backends import NumbaBackend
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class HabitatPatchElement(CompartmentElement):
    """Patch of space, a compartment of `n` organisms that divide and die at the same rate `alpha`.

    The divisions and deaths are jumps of `neutral_clustering`, so the number of organisms is a symmetric random walk,
    and an empty patch stays empty without arrivals from its neighbours.

    Args:
        label:
            The label of the patch or the lattice of patches.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        n:
            The initial number of organisms.

        alpha:
            The rate at which an organism divides, equal to the rate at which it dies.

    Examples:
        >>> from skcomplex.systems.biology.ecology import HabitatPatchElement
        >>> HabitatPatchElement('patch', n=5.0).alpha
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
        super().__init__(label, visual=visual, n=n)
        self.alpha = alpha


class HoppingInteraction(Interaction):
    """Hops of single organisms between neighbouring patches, each organism moving to each neighbour at the rate `beta`.

    The patches are the entries of a variadic role, and every pair `(i, j)` of `links` gives a jump that moves one
    organism from the patch of entry `i` to the patch of entry `j` at the rate `beta n_i`.

    Args:
        label:
            The label of the interaction.

        patches:
            The paths of the patches, such as the members `patches[i]` of a lattice.

        beta:
            The rate at which an organism hops to one neighbouring patch.

        links:
            The ordered pairs of entries of `patches` between which organisms hop.

    Examples:
        >>> from skcomplex.systems.biology.ecology import HoppingInteraction
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


def _neighbor_pairs(shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """Return the pairs of flat indices of every patch and each of its nearest neighbours on a periodic lattice."""
    indices = np.arange(int(np.prod(shape))).reshape(shape)
    pairs: list[tuple[int, int]] = []
    for axis in range(len(shape)):
        for step in (1, -1):
            neighbours = np.roll(indices, -step, axis=axis)
            pairs.extend(zip(indices.ravel().tolist(), neighbours.ravel().tolist(), strict=True))
    return pairs


@register_system(
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
            The system, with the numbers `patches.n` of organisms on the lattice `patches`, and their mean `mean` and
            variance `variance` over the patches.

    Examples:
        >>> from skcomplex.systems.biology.ecology import neutral_clustering
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
