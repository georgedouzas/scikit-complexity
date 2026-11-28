"""Coevolution of species whose fitnesses depend on each other.

The module holds the Bak-Sneppen system of punctuated equilibrium.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.laws.utils import minimum, neighbors, uniform, where
from ....core.components.spaces import (
    AdjacencyNeighborhood,
    BaseSpace,
    LatticeNeighborhood,
    LatticeSpace,
    NetworkSpace,
)
from ....core.engine.backends import BaseBackend, NumbaBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

SPECIES_NEIGHBORHOODS: dict[str, LatticeNeighborhood | AdjacencyNeighborhood] = {
    'ring': LatticeNeighborhood(((-1,), (1,))),
    'graph': AdjacencyNeighborhood(),
}
EXTINCTION_NEIGHBORS = ('ring', 'graph', 'random')
SPECIES_NEIGHBORS = ('fixed', 'random')


class SpeciesElement(Element):
    """Species with a fitness in the unit interval, one per site of a space.

    Args:
        label:
            The label of the species.

        fitness:
            The initial fitness, one per species or shared.

        space:
            The ring or the graph whose sites hold one species each, or `None` for one species.

    Examples:
        >>> from skcomplex.systems.biology.evolution import SpeciesElement
        >>> SpeciesElement('species', fitness=0.2).fitness
        0.2
    """

    visual = VisualSpec('image', {'states': ('fitness',), 'low': 0.0, 'high': 1.0, 'history': 256})

    def __init__(
        self,
        label: str,
        *,
        fitness: Annotated[float | Array, RealSet.closed_open(0, oo)] = 0.5,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.fitness = fitness


class ExtinctionInteraction(Interaction):
    """Extinctions of the Bak-Sneppen system, which renew the least fit species and its neighbours.

    Every step is one extinction. The least fit species of the group and its neighbours take new fitnesses, drawn
    uniformly from the unit interval, and every other species keeps its fitness. A species renews when the smallest
    fitness of the group lies in its neighbourhood or is its own, so the update needs no index of the least fit species.
    Ties of the smallest fitness, which uniform draws make vanishingly rare, renew every tied species with its
    neighbours. With `neighbors='random'` the neighbours are drawn afresh at every extinction: the least fit species
    and `k - 1` distinct other species drawn uniformly take new fitnesses, the random neighbour model whose fitnesses
    gather above `1 / k`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the species, whose value is `fitness`.

        neighbors:
            `'ring'` for the two neighbours on a ring, `'graph'` for the neighbours of a node, or `'random'` for species
            drawn afresh at every extinction.

        k:
            The number of species renewed at every extinction with random neighbours, the least fit one included,
            unused otherwise.

    Raises:
        ValidationError:
            The neighbours are unknown, or `k` is no whole number of at least 1.

    Examples:
        >>> from skcomplex.systems.biology.evolution import ExtinctionInteraction
        >>> ExtinctionInteraction('extinction', target='species').neighbors
        'ring'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role, neighbors: str = 'ring', k: int = 3) -> None:
        super().__init__(label)
        self.target = target
        self.neighbors = neighbors
        self.k = k

    def _renew_random(self, fitness: Array, *, random: np.random.Generator, **kwargs: object) -> Array:
        """Return the fitnesses after the least fit species and `k - 1` others drawn uniformly are renewed."""
        following = np.array(fitness, dtype=float)
        flat = following.reshape(-1)
        least = int(np.argmin(flat))
        others = random.choice(flat.size - 1, size=min(self.k - 1, flat.size - 1), replace=False)
        renewed = np.append(others + (others >= least), least)
        flat[renewed] = random.random(renewed.size)
        return following

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the update that renews the least fit species and its neighbours.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.

        Raises:
            ValidationError:
                The neighbours are unknown, or `k` is no whole number of at least 1.
        """
        super()._prepare(context)
        if self.neighbors not in EXTINCTION_NEIGHBORS:
            error_msg = f'Unknown neighbors {self.neighbors!r}. Use one of {", ".join(EXTINCTION_NEIGHBORS)}.'
            raise ValidationError(error_msg)
        if self.neighbors == 'random' and (int(self.k) != self.k or self.k < 1):
            error_msg = f'The extinction renews k={self.k!r} species. Give a whole number of at least 1.'
            raise ValidationError(error_msg)
        fitness = context.target.fitness
        if self.neighbors == 'random':
            self.laws_ = [UpdateLaw(fitness, self._renew_random, arrays=True, stochastic=True, label='extinction')]
            return
        around = neighbors(context.target.other.fitness, over=SPECIES_NEIGHBORHOODS[self.neighbors], reduce='min')
        renewed = minimum(fitness, around) == minimum(fitness)
        self.laws_ = [UpdateLaw(fitness, where(renewed, uniform('fitness'), fitness), label='extinction')]


class BakSneppenSystem(BaseSystem):
    """The Bak-Sneppen system of punctuated equilibrium.

    Its fitnesses self-organise above a critical value in avalanches of extinction.

    On a ring the fitnesses gather above the critical value 0.667, and the minimum fitness rises towards it through
    avalanches of every size. With `neighbors='random'` every extinction renews the least fit species and `k - 1`
    species drawn afresh, the random neighbour model of the mean field, whose fitnesses gather above `1 / k`. Every step
    is one extinction, so time counts the extinctions. The system has the fitnesses `species.fitness`.

    Args:
        size:
            The number of species on the ring, unused on a graph.

        graph:
            The graph whose nodes are the species, or `None` for a ring.

        neighbors:
            `'fixed'` for the neighbours on the ring or the graph, or `'random'` for species drawn afresh at every
            extinction.

        k:
            The number of species renewed at every extinction with random neighbours, the least fit one included,
            unused with fixed neighbours.

        random_state:
            The seed or generator of the initial fitnesses.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import BakSneppenSystem
        >>> BakSneppenSystem(50).prepare().system_.interactions[0].neighbors
        'ring'
        >>> fitness = BakSneppenSystem(50, neighbors='random', k=2).prepare().simulate(until=2000, random_state=0)
        >>> bool(fitness.get_values('species.fitness')[-1].min() > 0.3)
        True
    """

    field = 'biology/evolution'
    topics = ('lattice', 'network', 'self-organized criticality', 'extremal dynamics', 'stochastic')
    references = ('Bak and Sneppen 1993', 'Gros 2015', 'Jensen 2023')
    until = 400_000
    dt = 2_000

    def __init__(
        self,
        size: int = 200,
        *,
        graph: nx.Graph | None = None,
        neighbors: str = 'fixed',
        k: int = 3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.graph = graph
        self.neighbors = neighbors
        self.k = k
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Bak-Sneppen system.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The neighbours are neither `'fixed'` nor `'random'`.
        """
        super()._prepare(context)
        if self.neighbors not in SPECIES_NEIGHBORS:
            error_msg = f'BakSneppenSystem has the neighbours {self.neighbors!r}. Use one of {", ".join(SPECIES_NEIGHBORS)}.'
            raise ValidationError(error_msg)
        space = (LatticeSpace(self.size) if self.graph is None else NetworkSpace(self.graph)).prepare()
        count = int(space.size_ or 0)
        fitness = np.random.default_rng(self.random_state).random(count)
        species = SpeciesElement('species', fitness=fitness, space=space)
        extinction = ExtinctionInteraction(
            'extinction',
            target='species',
            neighbors=self.neighbors if self.neighbors == 'random' else 'ring' if self.graph is None else 'graph',
            k=self.k,
        )
        backend = NumbaBackend() if self.neighbors == 'fixed' else 'auto'
        self.system_ = System(self.label, elements=[species], interactions=[extinction], backend=backend)
