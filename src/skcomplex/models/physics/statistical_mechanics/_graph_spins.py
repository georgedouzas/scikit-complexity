"""Ising spins on the nodes of a graph, sampled by colored Monte Carlo sweeps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np

from ....core.base import Domain, parameter, state
from ....core.model import (
    Adjacency,
    Element,
    Energy,
    Law,
    Move,
    NetworkSpace,
    Sampler,
    Symbols,
    System,
    adjacent,
    neighbor_sum,
)
from ....core.runtime import register_model

NODES = 400
DEGREE = 4


class GraphIsingSpin(Element):
    """Ising spin on a node of a graph, with local energy `-J s sum(neighbours) - h s` over the node's edges.

    The sampler updates the classes of a proper coloring of the graph in turn, so no two neighbours flip together.

    Args:
        label:
            The label of the spin or the lattice group.

        spin:
            The initial spin, 1 or -1.

        J:
            The coupling between neighbouring spins.

        h:
            The external field.

        temperature:
            The temperature in units of the coupling.

        sampler:
            The acceptance rule, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import GraphIsingSpin
        >>> GraphIsingSpin('spin').sampler
        'metropolis'
    """

    spin: float = state(default=1.0, domain=Domain.INTEGER)
    J: float = parameter(default=1.0)
    h: float = parameter(default=0.0)
    temperature: float = parameter(default=2.0, domain=Domain.POSITIVE)
    sampler: str = 'metropolis'

    def laws(self, s: Symbols) -> list[Law]:
        """Return the neighbour-sum energy, the spin flip, and the colored sampler.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        energy = -s.J * s.spin * neighbor_sum(adjacent(s.spin)) - s.h * s.spin
        return [Energy(energy, Adjacency()), Move.flip(s.spin), Sampler(self.sampler, s.temperature, 'colored')]


@register_model(
    field='physics/statistical_mechanics',
    tags=('network', 'Monte Carlo', 'phase transition', 'spins'),
    references=('Argun et al. 2021', 'Dorogovtsev, Goltsev, and Mendes 2008'),
    defaults={'until': 200, 'dt': 1},
)
def graph_ising(
    graph: nx.Graph | None = None,
    *,
    temperature: float = 2.0,
    J: float = 1.0,
    h: float = 0.0,
    initial: str = 'random',
    sampler: str = 'metropolis',
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Ising model on a graph, ordered below a critical temperature set by the degrees of the nodes.

    On a random 4-regular graph the Bethe approximation gives the critical temperature `2 / ln 2`. One unit of time is
    one sweep of colored updates.

    Args:
        graph:
            The graph whose nodes hold the spins, or `None` for a random 4-regular graph of 400 nodes.

        temperature:
            The temperature in units of the coupling, which can change during a run.

        J:
            The coupling between neighbouring spins.

        h:
            The external field.

        initial:
            `'random'` for random spins or `'up'` for all spins up.

        sampler:
            The acceptance rule, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        random_state:
            The seed or generator of the default graph and the random initial spins.

    Returns:
        system:
            The model, with the spins `spins.spin` in node order.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.physics.statistical_mechanics import graph_ising
        >>> graph_ising(nx.cycle_graph(10)).elements[0].size
        10
    """
    rng = np.random.default_rng(random_state)
    if graph is None:
        graph = nx.random_regular_graph(DEGREE, NODES, seed=int(rng.integers(2**31)))
    size = graph.number_of_nodes()
    spins = np.ones(size) if initial == 'up' else rng.choice([-1.0, 1.0], size=size)
    space = NetworkSpace(graph)
    group = GraphIsingSpin.lattice('spins', space, spin=spins, J=J, h=h, temperature=temperature, sampler=sampler)
    return System(components=[group])
