"""Coevolution of species whose fitnesses depend on each other: the Bak-Sneppen model of punctuated equilibrium."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, LatticeElement
from ....core.laws import BaseLaw, ExtremalLaw
from ....core.neighborhoods import AdjacencyNeighborhood, LatticeNeighborhood
from ....core.spaces import LatticeSpace, NetworkSpace
from ....core.system import System
from ....specs import StateSpec, VisualSpec
from ....types import Array
from ..._catalog import register_model

SPECIES_NEIGHBORHOODS: dict[str, LatticeNeighborhood | AdjacencyNeighborhood] = {
    'ring': LatticeNeighborhood(((-1,), (1,))),
    'graph': AdjacencyNeighborhood(),
}
FITNESS_VISUAL = VisualSpec('image', {'states': ('fitness',), 'low': 0.0, 'high': 1.0, 'history': 256})
FITNESS_SITES = VisualSpec('sites', {'states': ('fitness',), 'low': 0.0, 'high': 1.0})


def _renew(fitness: Array, random: Array) -> Array:
    """Return new fitnesses drawn uniformly from the unit interval for the least fit species and its neighbours."""
    return random


class SpeciesElement(Element):
    """Species of the Bak-Sneppen model, with a fitness in the unit interval that depends on its neighbours.

    Every update replaces the fitness of the least fit species and of its neighbours by new uniform random numbers, and
    is recorded as an `extinction` event with the replaced species and its fitness. One step holds `updates` updates.

    Args:
        label:
            The label of the species or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        fitness:
            The initial fitness.

        updates:
            The number of updates per step.

        neighbors:
            `'ring'` for the two neighbours on a ring or `'graph'` for the neighbours of a node.

    Examples:
        >>> from skcomplex.models.biology.evolution import SpeciesElement
        >>> SpeciesElement('species').updates
        200
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        updates: int = 200,
        neighbors: str = 'ring',
    ) -> None:
        super().__init__(label, visual=visual)
        self.fitness = fitness
        self.updates = updates
        self.neighbors = neighbors

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the extremal update of the least fit species and its neighbours.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        neighborhood = SPECIES_NEIGHBORHOODS[self.neighbors]
        return [ExtremalLaw(s.fitness, neighborhood, _renew, repeat=self.updates, label='extinction')]


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
    space = (LatticeSpace(size) if graph is None else NetworkSpace(graph)).check().prepare()
    count = int(space.size_ or 0)
    fitness = np.random.default_rng(random_state).random(count)
    species = LatticeElement(
        'species',
        element=SpeciesElement,
        space=space,
        values={
            'fitness': fitness,
            'updates': count if updates is None else updates,
            'neighbors': 'ring' if graph is None else 'graph',
        },
        visual=FITNESS_VISUAL if graph is None else FITNESS_SITES,
    )
    return System(elements=[species])
