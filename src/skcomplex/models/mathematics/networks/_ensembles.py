"""Random graph ensembles and the robustness of networks to failures and attacks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

import networkx as nx
import numpy as np

from ....core.elements import Element, GroupElement
from ....core.interactions import MapInteraction, TopologyInteraction
from ....core.laws.utils import maximum, neighbors, where
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ....errors import ValidationError
from ..._catalog import register_model
from ._ensemble_node import EnsembleNodeElement
from ._failing import FailingElement
from ._link import LinkElement
from ._vertex import DEGREE_VISUAL

ENSEMBLES = ('erdos_renyi', 'watts_strogatz')


STRATEGIES = ('random', 'targeted')


def _get_removal(s: SimpleNamespace) -> object:
    """Return the probability that a node is removed in a step, at random or as a node of the largest degree."""
    degree = neighbors(1)
    hub = where(s.source.targeted * (degree - maximum(degree)) >= 0, s.source.targeted, 0)
    return (1 - s.source.targeted) * s.source.rate + hub


def _get_degree() -> MapInteraction:
    """Return the map that sets the degree of every node from its edges at the start of a step."""
    return MapInteraction('degree', target='nodes', state='degree', function=lambda s: neighbors(1))


def _variable_network(
    element: type[Element],
    space: NetworkSpace,
    capacity: int,
    **values: object,
) -> list[GroupElement]:
    """Return the variable groups on the nodes and the edges of a network."""
    nodes = GroupElement(
        'nodes',
        element=element,
        on='sites',
        variable=True,
        capacity=space.prepare().size_,
        values=values,
        visual=DEGREE_VISUAL,
    )
    links = GroupElement('links', element=LinkElement, on='edges', variable=True, capacity=capacity)
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
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.mathematics.networks import network_attack
        >>> trajectory = Simulation(network_attack(50), random_state=0).prepare().run(until=3).trajectory_
        >>> int(trajectory.network(-1).nodes.size) < 50
        True
    """
    if strategy not in STRATEGIES:
        error_msg = f'Unknown strategy {strategy!r}. Use one of {", ".join(STRATEGIES)}.'
        raise ValidationError(error_msg)
    space = NetworkSpace(nx.barabasi_albert_graph(n_nodes, edges, seed=random_state))
    targeted = float(strategy == 'targeted')
    groups = _variable_network(FailingElement, space, space.graph.number_of_edges(), rate=rate, targeted=targeted)
    removal = TopologyInteraction('removal', target='nodes', action='remove-node', probability=_get_removal)
    return System(elements=groups, interactions=[_get_degree(), removal], space=space, time=DiscreteTime())


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
            The model, with the groups `nodes` and `links`, the map `degree`, and the interaction `linking` or
            `rewiring`.

    Raises:
        ValidationError:
            The ensemble is unknown.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.mathematics.networks import random_graph_ensembles
        >>> system = random_graph_ensembles(40, ensemble='watts_strogatz', steps=5)
        >>> trajectory = Simulation(system, random_state=0).prepare().run(until=5).trajectory_
        >>> int(trajectory.network(-1).sources.size)
        80
    """
    if ensemble not in ENSEMBLES:
        error_msg = f'Unknown ensemble {ensemble!r}. Use one of {", ".join(ENSEMBLES)}.'
        raise ValidationError(error_msg)
    if ensemble == 'erdos_renyi':
        space = NetworkSpace(nx.empty_graph(n_nodes))
        rate = mean_degree / 2 / steps
        groups = _variable_network(
            EnsembleNodeElement,
            space,
            int(np.ceil(1.5 * mean_degree * n_nodes / 2)) + 10,
            rate=rate,
        )
        linking = TopologyInteraction('linking', target='nodes', action='add-edge', probability=lambda s: s.source.rate)
        return System(elements=groups, interactions=[_get_degree(), linking], space=space, time=DiscreteTime())
    space = NetworkSpace(nx.watts_strogatz_graph(n_nodes, int(mean_degree), 0.0))
    rate = 1 - (1 - rewiring) ** (1 / steps)
    groups = _variable_network(EnsembleNodeElement, space, space.graph.number_of_edges())
    rewiring = TopologyInteraction('rewiring', target='nodes', action='rewire', probability=rate)
    return System(elements=groups, interactions=[_get_degree(), rewiring], space=space, time=DiscreteTime())
