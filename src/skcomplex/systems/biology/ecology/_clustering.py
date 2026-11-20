"""Organisms that divide, die, and hop between patches, and gather in clusters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import JumpLaw
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend, NumbaBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.populations import NeutralBirthInteraction, NeutralDeathInteraction


class HabitatPatchElement(Element):
    """Patch of space, a compartment of `x` organisms that divide and die.

    The divisions and deaths of `NeutralClusteringSystem` happen at the same rate, so the number of organisms is a
    symmetric random walk, and an empty patch stays empty without arrivals from its neighbours.

    Args:
        label:
            The label of the patch or of the lattice of patches.

        x:
            The initial number of organisms.

        size:
            The number of patches, or `None` for one.

        space:
            The lattice with one patch per site, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.ecology import HabitatPatchElement
        >>> HabitatPatchElement('patch', x=5.0).x
        5.0
    """

    visual = VisualSpec('image', {'low': 0.0})

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float | npt.ArrayLike, NN] = 10.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x


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
        patches: tuple[Role, ...],
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        links: Sequence[tuple[int, int]] = (),
    ) -> None:
        super().__init__(label)
        self.patches = patches
        self.beta = beta
        self.links = links

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set one hop per ordered pair of neighbouring patches.

        Args:
            context:
                The namespace of the bound elements, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            patches = s.patches
            self.laws_ = [JumpLaw(s.beta * patches[i].x, {patches[i].x: -1, patches[j].x: 1}) for i, j in self.links]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.ecology import NeutralClusteringSystem
        >>> len(NeutralClusteringSystem(8).prepare().system_.elements[0].x)
        8
    """

    field = 'biology/ecology'
    topics = ('neutral theory', 'clustering', 'jump process', 'lattice', 'birth-death', 'diffusion')
    references = ('Bertin 2021', 'Houchmandzadeh 2002', 'Houchmandzadeh 2009')
    until = 50.0
    dt = 0.5
    observables = (
        Observable('mean', group='patches', variable='x', reduce='mean'),
        Observable('variance', group='patches', variable='x', reduce='var'),
    )

    def __init__(
        self,
        shape: int | Sequence[int] = 64,
        *,
        n: float = 10.0,
        alpha: float = 1.0,
        beta: float = 0.5,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of organisms that divide.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        lattice = (self.shape,) if isinstance(self.shape, int) else tuple(self.shape)
        patches = HabitatPatchElement('patches', x=np.full(lattice, float(self.n)), space=LatticeSpace(lattice))
        members = [f'patches[{index}]' for index in range(int(np.prod(lattice)))]
        interactions: list[BaseInteraction] = [
            NeutralBirthInteraction('division', target='patches', b0=0.0, b1=self.alpha),
            NeutralDeathInteraction('death', target='patches', d0=0.0, d1=self.alpha),
        ]
        if self.beta:
            interactions.append(
                HoppingInteraction(
                    'hopping',
                    patches=tuple(members),
                    links=self._get_neighbor_pairs(lattice),
                    beta=self.beta,
                ),
            )
        self.system_ = System(
            self.label,
            elements=[patches],
            interactions=interactions,
            backend=NumbaBackend(),
        )
