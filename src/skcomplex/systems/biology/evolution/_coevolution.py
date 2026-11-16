"""Coevolution of species whose fitnesses depend on each other.

The module holds the Bak-Sneppen system of punctuated equilibrium.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import minimum, neighbors, uniform, where
from ....core.components.spaces import AdjacencyNeighborhood, LatticeNeighborhood, LatticeSpace, NetworkSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.backends import NumbaBackend
from ....core.specs import StateSpec, VisualSpec

SPECIES_NEIGHBORHOODS: dict[str, LatticeNeighborhood | AdjacencyNeighborhood] = {
    'ring': LatticeNeighborhood(((-1,), (1,))),
    'graph': AdjacencyNeighborhood(),
}


class SpeciesElement(Element):
    """Species of the Bak-Sneppen system, with a fitness in the unit interval.

    The fitness depends on the neighbours of the species.

    Every step is one extinction. The least fit species of the group and its neighbours take new fitnesses, drawn
    uniformly from the unit interval, and every other species keeps its fitness. A species renews when the smallest
    fitness of the group lies in its neighbourhood or is its own, so the update needs no index of the least fit species.
    Ties of the smallest fitness, which uniform draws make vanishingly rare, renew every tied species with its
    neighbours.

    Args:
        label:
            The label of the species or the lattice group.

        fitness:
            The initial fitness.

        neighbors:
            `'ring'` for the two neighbours on a ring or `'graph'` for the neighbours of a node.

    Examples:
        >>> from skcomplex.systems.biology.evolution import SpeciesElement
        >>> SpeciesElement('species').neighbors
        'ring'
    """

    visual = VisualSpec('image', {'states': ('fitness',), 'low': 0.0, 'high': 1.0, 'history': 256})
    law_classes = (UpdateLaw,)
    draws = True

    def __init__(
        self,
        label: str,
        *,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        neighbors: str = 'ring',
    ) -> None:
        super().__init__(label)
        self.fitness = fitness
        self.neighbors = neighbors

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the update that renews the least fit species and its neighbours.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        around = neighbors(s.other.fitness, over=SPECIES_NEIGHBORHOODS[self.neighbors], reduce='min')
        renewed = minimum(s.fitness, around) == minimum(s.fitness)
        return [UpdateLaw(s.fitness, where(renewed, uniform('fitness'), s.fitness), label='extinction')]


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

    Examples:
        >>> from skcomplex.systems.biology.evolution import BakSneppenSystem
        >>> BakSneppenSystem(50).prepare().system_.elements[0].values['neighbors']
        'ring'
    """

    field = 'biology/evolution'
    topics = ('lattice', 'network', 'self-organized criticality', 'extremal dynamics', 'stochastic')
    references = ('Bak and Sneppen 1993', 'Gros 2015', 'Jensen 2023')
    until = 400_000
    dt = 2_000
    backend = NumbaBackend()

    def __init__(
        self,
        size: int = 200,
        *,
        graph: nx.Graph | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.graph = graph
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the Bak-Sneppen system."""
        space = (LatticeSpace(self.size) if self.graph is None else NetworkSpace(self.graph)).prepare()
        count = int(space.size_ or 0)
        fitness = np.random.default_rng(self.random_state).random(count)
        species = GroupElement(
            'species',
            element=SpeciesElement,
            space=space,
            values={
                'fitness': fitness,
                'neighbors': 'ring' if self.graph is None else 'graph',
            },
            on='sites',
        )
        return CompositeSystem(self.label, elements=[species])
