"""Coevolution of species whose fitnesses depend on each other: the Bak-Sneppen model of punctuated equilibrium."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, state
from ....core.components import Element
from ....core.laws import Extremal, Law, Symbols
from ....core.runtime import register_model
from ....core.space import Adjacency, LatticeSpace, Neighborhood, NetworkSpace
from ....core.system import System

type Array = npt.NDArray[np.float64]

SPECIES_NEIGHBORHOODS: dict[str, Neighborhood | Adjacency] = {'ring': Neighborhood(((-1,), (1,))), 'graph': Adjacency()}
FITNESS_VISUAL = VisualSpec('image', {'states': ('fitness',), 'low': 0.0, 'high': 1.0, 'history': 256})
FITNESS_SITES = VisualSpec('sites', {'states': ('fitness',), 'low': 0.0, 'high': 1.0})


def _renew(fitness: Array, random: Array) -> Array:
    """Return new fitnesses drawn uniformly from the unit interval for the least fit species and its neighbours."""
    return random


class Species(Element):
    """Species of the Bak-Sneppen model, with a fitness in the unit interval that depends on its neighbours.

    Every update replaces the fitness of the least fit species and of its neighbours by new uniform random numbers, and
    is recorded as an `extinction` event with the replaced species and its fitness. One step holds `updates` updates.

    Args:
        label:
            The label of the species or the lattice group.

        fitness:
            The initial fitness.

        updates:
            The number of updates per step.

        neighbors:
            `'ring'` for the two neighbours on a ring or `'graph'` for the neighbours of a node.

    Examples:
        >>> from skcomplex.models.biology.evolution import Species
        >>> Species('species').updates
        200
    """

    fitness: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    updates: int = 200
    neighbors: str = 'ring'

    def laws(self, s: Symbols) -> list[Law]:
        """Return the extremal update of the least fit species and its neighbours.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        neighborhood = SPECIES_NEIGHBORHOODS[self.neighbors]
        return [Extremal(s.fitness, neighborhood, _renew, repeat=self.updates, label='extinction')]


@register_model(
    field='biology/evolution',
    tags=('lattice', 'network', 'self-organized criticality', 'extremal dynamics', 'stochastic'),
    references=('Bak and Sneppen 1993', 'Gros 2015', 'Jensen 2023'),
    defaults={'until': 2000, 'dt': 10},
)
def bak_sneppen(
    size: int = 200,
    *,
    graph: nx.Graph | None = None,
    updates: int | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Bak-Sneppen model, whose fitnesses self-organise above a critical value with avalanches of extinction.

    On a ring the fitnesses gather above the critical value 0.667, and the minimum fitness rises towards it through
    avalanches of every size.

    Args:
        size:
            The number of species on the ring, unused on a graph.

        graph:
            The graph whose nodes are the species, or `None` for a ring.

        updates:
            The number of updates per step, or `None` for one update per species.

        random_state:
            The seed or generator of the initial fitnesses.

    Returns:
        system:
            The model, with the fitnesses `species.fitness`, and `extinction` events with the site and the minimum
            fitness of every update.

    Examples:
        >>> from skcomplex.models.biology.evolution import bak_sneppen
        >>> bak_sneppen(50).elements[0].values['updates']
        50
    """
    space = LatticeSpace(size) if graph is None else NetworkSpace(graph)
    count = space.size if isinstance(space, LatticeSpace) else space.graph.number_of_nodes()
    fitness = np.random.default_rng(random_state).random(count)
    species = Species.lattice(
        'species',
        space,
        visual=FITNESS_VISUAL if graph is None else FITNESS_SITES,
        fitness=fitness,
        updates=count if updates is None else updates,
        neighbors='ring' if graph is None else 'graph',
    )
    return System(components=[species])
