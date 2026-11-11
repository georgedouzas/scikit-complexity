"""Organisms that divide, die, and hop between patches, and gather in clusters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement
from ....core.components.interactions import BaseInteraction, Interaction, JumpInteraction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.backends import NumbaBackend
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.kinetics import CompartmentElement


class HabitatPatchElement(CompartmentElement):
    """Patch of space, a compartment of `x` organisms that divide and die.

    The organisms divide and die at the same rate `alpha`.

    The divisions and deaths are jumps of `NeutralClusteringSystem`, so the number of organisms is a symmetric random
    walk, and an empty patch stays empty without arrivals from its neighbours.

    Args:
        label:
            The label of the patch or the lattice of patches.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial number of organisms.

        alpha:
            The rate at which an organism divides, equal to the rate at which it dies.

    Examples:
        >>> from skcomplex.systems.biology.ecology import HabitatPatchElement
        >>> HabitatPatchElement('patch', x=5.0).alpha
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('image', {'low': 0.0}),
        *,
        x: Annotated[float, StateSpec(domain=NN)] = 10.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual, x=x)
        self.alpha = alpha


class HoppingInteraction(Interaction):
    """Hops of single organisms between neighbouring patches.

    Each organism moves to each neighbour at the rate `beta`.

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

    law_classes = (JumpLaw,)

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
        return [JumpLaw(s.beta * patches[i].x, {patches[i].x: -1, patches[j].x: 1}) for i, j in self.links]


class NeutralClusteringSystem(BaseSystem):
    """Organisms that divide, die, and hop between patches on a periodic lattice.

    They gather in clusters.

    Births balance deaths, so the mean number of organisms per patch stays `n` while its variance grows. Without hops it
    grows as `2 alpha n t`, so most patches empty while a few fill. Hops at the rate `beta` slow the growth to `t^(1/2)`
    in one dimension and `ln t` in two, so the clusters persist in low dimension although the mean density obeys a
    diffusion equation. The system has the numbers `patches.x` of organisms on the lattice `patches`, and their mean
    `mean` and variance `variance` over the patches.

    Args:
        shape:
            The number of patches of a ring, or the shape of a periodic lattice of patches.

        n:
            The initial number of organisms in every patch.

        alpha:
            The rate at which an organism divides, equal to the rate at which it dies.

        beta:
            The rate at which an organism hops to one neighbouring patch.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import NeutralClusteringSystem
        >>> NeutralClusteringSystem(8).prepare().system_.elements[0].prepare().size_
        8
    """

    field = 'biology/ecology'
    topics = ('neutral theory', 'clustering', 'jump process', 'lattice', 'birth-death', 'diffusion')
    references = ('Bertin 2021', 'Houchmandzadeh 2002', 'Houchmandzadeh 2009')
    until = 50.0
    dt = 0.5
    backend = NumbaBackend()
    observables = (
        ObservableSpec('mean', group='patches', variable='x', reduce='mean'),
        ObservableSpec('variance', group='patches', variable='x', reduce='var'),
    )

    def __init__(
        self,
        shape: int | Sequence[int] = 64,
        *,
        n: float = 10.0,
        alpha: float = 1.0,
        beta: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.n = n
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _get_neighbor_pairs(shape: tuple[int, ...]) -> list[tuple[int, int]]:
        """Return the pairs of flat indices of every patch and its nearest neighbours.

        The lattice is periodic.
        """
        indices = np.arange(int(np.prod(shape))).reshape(shape)
        pairs: list[tuple[int, int]] = []
        for axis in range(len(shape)):
            for step in (1, -1):
                neighbours = np.roll(indices, -step, axis=axis)
                pairs.extend(zip(indices.ravel().tolist(), neighbours.ravel().tolist(), strict=True))
        return pairs

    def _build_system(self) -> CompositeSystem:
        """Return the components of organisms that divide."""
        lattice = (self.shape,) if isinstance(self.shape, int) else tuple(self.shape)
        patches = GroupElement(
            'patches',
            element=HabitatPatchElement,
            space=LatticeSpace(lattice),
            values={'x': np.full(lattice, float(self.n)), 'alpha': self.alpha},
            on='sites',
        )
        members = [f'patches[{index}]' for index in range(int(np.prod(lattice)))]
        interactions: list[BaseInteraction] = [
            JumpInteraction('division', target='patches', changes={'x': 1}, rate=lambda s: s.target.alpha * s.target.x),
            JumpInteraction('death', target='patches', changes={'x': -1}, rate=lambda s: s.target.alpha * s.target.x),
        ]
        if self.beta:
            interactions.append(
                HoppingInteraction('hopping', patches=members, links=self._get_neighbor_pairs(lattice), beta=self.beta),
            )
        return CompositeSystem(
            self.label,
            elements=[patches],
            interactions=interactions,
        )
