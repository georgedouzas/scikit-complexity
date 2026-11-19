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
from ....core.components.elements import Element
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
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

SPECIES_NEIGHBORHOODS: dict[str, LatticeNeighborhood | AdjacencyNeighborhood] = {
    'ring': LatticeNeighborhood(((-1,), (1,))),
    'graph': AdjacencyNeighborhood(),
}


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
    neighbours.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the species, whose value is `fitness`.

        neighbors:
            `'ring'` for the two neighbours on a ring or `'graph'` for the neighbours of a node.

    Examples:
        >>> from skcomplex.systems.biology.evolution import ExtinctionInteraction
        >>> ExtinctionInteraction('extinction', target='species').neighbors
        'ring'
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role, neighbors: str = 'ring') -> None:
        super().__init__(label)
        self.target = target
        self.neighbors = neighbors

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the update that renews the least fit species and its neighbours.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            fitness = context.target.fitness
            around = neighbors(context.target.other.fitness, over=SPECIES_NEIGHBORHOODS[self.neighbors], reduce='min')
            renewed = minimum(fitness, around) == minimum(fitness)
            self.laws_ = [UpdateLaw(fitness, where(renewed, uniform('fitness'), fitness), label='extinction')]


class BakSneppenSystem(BaseSystem):
    """The Bak-Sneppen system of punctuated equilibrium.

    Its fitnesses self-organise above a critical value in avalanches of extinction.

    On a ring the fitnesses gather above the critical value 0.667, and the minimum fitness rises towards it through
    avalanches of every size. Every step is one extinction, so time counts the extinctions. The system has the
    fitnesses `species.fitness`.

    Args:
        size:
            The number of species on the ring, unused on a graph.

        graph:
            The graph whose nodes are the species, or `None` for a ring.

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
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.graph = graph
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the Bak-Sneppen system.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        space = (LatticeSpace(self.size) if self.graph is None else NetworkSpace(self.graph)).prepare()
        count = int(space.size_ or 0)
        fitness = np.random.default_rng(self.random_state).random(count)
        species = SpeciesElement('species', fitness=fitness, space=space)
        extinction = ExtinctionInteraction(
            'extinction',
            target='species',
            neighbors='ring' if self.graph is None else 'graph',
        )
        self.system_ = System(self.label, elements=[species], interactions=[extinction], backend=NumbaBackend())
