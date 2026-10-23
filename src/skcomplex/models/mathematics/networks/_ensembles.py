"""Random graph ensembles and the robustness of networks to failures and attacks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import networkx as nx
import numpy as np

from ....core.base import Domain, ValidationError, VisualSpec, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import Law, Symbols, Update, add_edge, maximum, neighbors, remove_node, rewire, where
from ....core.runtime import register_model
from ....core.space import DiscreteTime, NetworkSpace
from ....core.system import System
from ._growth import DEGREE_VISUAL, Link

ENSEMBLES = ('erdos_renyi', 'watts_strogatz')
STRATEGIES = ('random', 'targeted')


class Failing(Element):
    """Node of a network under failures or attacks, removed at random or when it has the largest degree.

    Every step a node fails with probability `rate` under random failures, and under a targeted attack the nodes
    of the largest degree are removed, one degree class per step.

    Args:
        label:
            The label of the node group.

        degree:
            The number of edges of the node, read at the start of every step.

        rate:
            The probability that a node fails in a step under random failures.

        targeted:
            1 for a targeted attack on the nodes of the largest degree and 0 for random failures.

    Examples:
        >>> from skcomplex.models.mathematics.networks import Failing
        >>> Failing('nodes', targeted=1).targeted
        1
    """

    visual: ClassVar[VisualSpec] = DEGREE_VISUAL
    degree: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    rate: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    targeted: float = parameter(default=0, domain=Domain.BOOLEAN)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the degree of the node and its removal.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        degree = neighbors(1)
        hub = where(s.targeted * (degree - maximum(degree)) >= 0, s.targeted, 0)
        return [Update(s.degree, degree), Update(remove_node(), (1 - s.targeted) * s.rate + hub)]


class EnsembleNode(Element):
    """Node of a random graph that adds edges to uniformly random nodes, building an Erdős-Rényi graph edge by edge.

    Args:
        label:
            The label of the node group.

        degree:
            The number of edges of the node, read at the start of every step.

        rate:
            The probability that the node adds an edge to a random node in a step.

    Examples:
        >>> from skcomplex.models.mathematics.networks import EnsembleNode
        >>> EnsembleNode('nodes', rate=0.1).rate
        0.1
    """

    visual: ClassVar[VisualSpec] = DEGREE_VISUAL
    degree: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    rate: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the degree of the node and its new edges.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Update(s.degree, neighbors(1)), Update(add_edge(), s.rate)]


class Rewiring(Interaction):
    """Rewiring of every edge of a ring lattice to a uniformly random node, the Watts-Strogatz construction.

    Every step every edge moves its target to a random node with probability `rate`, so after `n` steps an edge has
    moved with probability `1 - (1 - rate)^n`. A second move draws a new random end, so the result follows the
    Watts-Strogatz ensemble of that rewiring probability.

    Args:
        label:
            The label of the interaction.

        nodes:
            The path of the group on the nodes.

        links:
            The path of the variable group on the edges.

        rate:
            The probability that an edge moves in a step.

    Examples:
        >>> from skcomplex.models.mathematics.networks import Rewiring
        >>> Rewiring('rewiring', nodes='nodes', links='links', rate=0.01).rate
        0.01
    """

    over: ClassVar[str] = 'neighbours'
    nodes: str = role()
    links: str = role()
    rate: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rewiring of one edge.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Update(rewire(keep='source'), s.rate)]


def _variable_network(element: type[Element], space: NetworkSpace, capacity: int, **values: object) -> list[Group]:
    """Return the variable groups on the nodes and the edges of a network."""
    nodes = Group(
        'nodes',
        element=element,
        on='sites',
        variable=True,
        capacity=space.size,
        values=values,
        visual=DEGREE_VISUAL,
    )
    links = Group('links', element=Link, on='edges', variable=True, capacity=capacity)
    return [nodes, links]


@register_model(
    field='mathematics/networks',
    tags=('network', 'robustness', 'percolation', 'attack', 'scale-free'),
    references=('Albert, Jeong, and Barabási 2000', 'Cohen et al. 2000', 'Ramasco 2020'),
    defaults={'until': 100},
)
def network_attack(
    n_nodes: int = 300,
    *,
    edges: int = 2,
    strategy: str = 'targeted',
    rate: float = 0.01,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return failures or attacks on a scale-free network, robust to random failures but fragile to attacks on hubs.

    The network is a Barabási-Albert graph. Under random failures every node fails with probability `rate` per step,
    and the giant component shrinks slowly. Under a targeted attack the nodes of the largest degree are removed
    every step, and the giant component breaks up after a few percent of the nodes are gone.

    Args:
        n_nodes:
            The number of nodes.

        edges:
            The number of edges every node of the Barabási-Albert graph joins with.

        strategy:
            `'random'` for random failures or `'targeted'` for attacks on the largest degree.

        rate:
            The probability that a node fails in a step under random failures.

        random_state:
            The seed or generator of the network.

    Returns:
        system:
            The model, with the groups `nodes` and `links`.

    Raises:
        ValidationError:
            The strategy is unknown.

    Examples:
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.models.mathematics.networks import network_attack
        >>> trajectory = Simulation(network_attack(50), random_state=0).run(until=3).trajectory_
        >>> int(trajectory.network(-1).nodes.size) < 50
        True
    """
    if strategy not in STRATEGIES:
        error_msg = f'Unknown strategy {strategy!r}. Use one of {", ".join(STRATEGIES)}.'
        raise ValidationError(error_msg)
    space = NetworkSpace.barabasi_albert(n_nodes, edges, seed=random_state)
    targeted = float(strategy == 'targeted')
    groups = _variable_network(Failing, space, space.graph.number_of_edges(), rate=rate, targeted=targeted)
    return System(components=groups, space=space, time=DiscreteTime())


@register_model(
    field='mathematics/networks',
    tags=('network', 'random graph', 'small world', 'giant component'),
    references=('Erdős and Rényi 1959', 'Watts and Strogatz 1998', 'Barabási 2016'),
    defaults={'until': 100},
)
def random_graph_ensembles(
    n_nodes: int = 200,
    *,
    ensemble: str = 'erdos_renyi',
    mean_degree: float = 4.0,
    rewiring: float = 0.2,
    steps: int = 100,
) -> System:
    """Return the construction of an Erdős-Rényi or a Watts-Strogatz random graph, one step of new edges at a time.

    The Erdős-Rényi graph starts empty and every node adds an edge to a random node with a probability that brings
    the mean degree to `mean_degree` after `steps` steps, so a giant component emerges when the mean degree passes 1.
    The Watts-Strogatz graph starts as a ring where every node joins its `mean_degree` nearest nodes, and every edge
    moves its target to a random node at a rate that rewires a fraction `rewiring` of the edges after `steps` steps,
    which shortens paths while the clustering stays high.

    Args:
        n_nodes:
            The number of nodes.

        ensemble:
            `'erdos_renyi'` or `'watts_strogatz'`.

        mean_degree:
            The mean degree of the final graph, an even number for the Watts-Strogatz ring.

        rewiring:
            The fraction of the edges of the Watts-Strogatz ring rewired after `steps` steps.

        steps:
            The number of steps of the construction.

    Returns:
        system:
            The model, with the groups `nodes` and `links` and, for Watts-Strogatz, the interaction `rewiring`.

    Raises:
        ValidationError:
            The ensemble is unknown.

    Examples:
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.models.mathematics.networks import random_graph_ensembles
        >>> system = random_graph_ensembles(40, ensemble='watts_strogatz', steps=5)
        >>> trajectory = Simulation(system, random_state=0).run(until=5).trajectory_
        >>> int(trajectory.network(-1).sources.size)
        80
    """
    if ensemble not in ENSEMBLES:
        error_msg = f'Unknown ensemble {ensemble!r}. Use one of {", ".join(ENSEMBLES)}.'
        raise ValidationError(error_msg)
    if ensemble == 'erdos_renyi':
        space = NetworkSpace(nx.empty_graph(n_nodes))
        rate = mean_degree / 2 / steps
        groups = _variable_network(EnsembleNode, space, int(np.ceil(1.5 * mean_degree * n_nodes / 2)) + 10, rate=rate)
        return System(components=groups, space=space, time=DiscreteTime())
    space = NetworkSpace.watts_strogatz(n_nodes, int(mean_degree), 0.0)
    rate = 1 - (1 - rewiring) ** (1 / steps)
    groups = _variable_network(EnsembleNode, space, space.graph.number_of_edges())
    rewiring_law = Rewiring('rewiring', nodes='nodes', links='links', rate=rate)
    return System(components=groups, interactions=[rewiring_law], space=space, time=DiscreteTime())
