"""Synchronization on complex networks: Kuramoto oscillators and coupled logistic maps on the edges of a graph."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement, LatticeElement
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import neighbors
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ..common.oscillators import KuramotoCouplingInteraction, PhaseOscillatorElement
from ._utils import _order_parameter

NETWORKS = ('watts_strogatz', 'erdos_renyi', 'barabasi_albert')


class LinkElement(Element):
    """Edge of a network with a weight `w`, which couplings along the edge read.

    Args:
        label:
            The label of the edge or the group on the edges.

        visual:
            How the edge is drawn, or `None` for `default_visual_`.

        w:
            The weight of the edge.

    Attributes:
        default_visual_:
            The visual of the edge when it sets none.

    Examples:
        >>> from skcomplex.models.physics.synchronization import LinkElement
        >>> LinkElement('link', w=0.5).w
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        w: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.w = w


class LogisticNodeElement(Element):
    """Logistic map on a node of a network, `x_i(t + 1) = (1 - epsilon) f(x_i) + epsilon mean_j f(x_j)`.

    The map `f(x) = r x (1 - x)` of the node mixes with the mean of the maps of its neighbours `j` on the network. A
    node with no neighbours keeps `(1 - epsilon) f(x_i)`.

    Args:
        label:
            The label of the node or the group on the nodes.

        visual:
            How the node is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        r:
            The growth rate of the logistic map.

        epsilon:
            The weight of the neighbours.

    Attributes:
        default_visual_:
            The visual of the node when it sets none.

    Examples:
        >>> from skcomplex.models.physics.synchronization import LogisticNodeElement
        >>> LogisticNodeElement('node').epsilon
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.5,
        r: Annotated[float, ParameterSpec()] = 4.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r
        self.epsilon = epsilon

    def _logistic(self, r: Expression, x: Expression) -> Expression:
        """Return the logistic map `r x (1 - x)`."""
        return r * x * (1 - x)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the value to the mix of its map and the mean map of the neighbours.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the neighbour `namespace.other`.

        Returns:
            laws:
                The update of the value.
        """
        s = namespace
        own, others = self._logistic(s.r, s.x), neighbors(self._logistic(s.r, s.other.x), reduce='mean')
        return [UpdateLaw(s.x, (1 - s.epsilon) * own + s.epsilon * others)]


class EdgeCouplingInteraction(KuramotoCouplingInteraction):
    """Kuramoto coupling along the weighted edges of a network, `theta_i' += K w_ij sin(theta_j - theta_i)`.

    The coupling runs once per edge of the network of its oscillators and reads the weight `w` of the edge from the
    group on the edges.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of oscillators on the nodes.

        links:
            The path of the group on the edges, whose parameter `w` weighs every edge.

        K:
            The coupling strength.

        alpha:
            The phase lag.

    Examples:
        >>> from skcomplex.models.physics.synchronization import EdgeCouplingInteraction
        >>> EdgeCouplingInteraction('coupling', target='nodes', links='edges', K=0.5).weights
        'w'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        K: float = 1.0,
        alpha: float = 0.0,
    ) -> None:
        super().__init__(label, target=target, K=K, alpha=alpha, weights='w')
        self.links = links


def _network(name: str, n_nodes: int, degree: int, rewiring: float, seed: int) -> NetworkSpace:
    """Return a network of a generator with a mean degree near `degree`."""
    rng = np.random.default_rng(seed)
    if name == 'watts_strogatz':
        return NetworkSpace(nx.watts_strogatz_graph(n_nodes, degree, rewiring, seed=rng))
    if name == 'erdos_renyi':
        return NetworkSpace(nx.gnp_random_graph(n_nodes, degree / (n_nodes - 1), seed=rng))
    if name == 'barabasi_albert':
        return NetworkSpace(nx.barabasi_albert_graph(n_nodes, max(degree // 2, 1), seed=rng))
    error_msg = f'Unknown network {name!r}. Use one of {", ".join(NETWORKS)}.'
    raise ValueError(error_msg)


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'oscillators', 'phase', 'network'),
    references=('Arenas et al. 2008', 'Jensen 2023'),
    defaults={'until': 40.0, 'dt': 0.2},
    observables=lambda system: _order_parameter('order', 'oscillators'),
)
def kuramoto_on_networks(
    n_oscillators: int = 200,
    *,
    network: str = 'watts_strogatz',
    degree: int = 6,
    rewiring: float = 0.1,
    K: float = 1.0,
    sigma: float = 0.5,
    seed: int = 0,
) -> System:
    """Return Kuramoto oscillators coupled along the edges of a network, `theta_i' = omega_i + K sum_j A_ij sin(...)`.

    Oscillator `i` follows `theta_i' = omega_i + (K / <k>) sum_j w_ij sin(theta_j - theta_i)` over its neighbours `j`,
    where the weights `w_ij = 1` live on a group on the edges and `<k>` is the mean degree. The coupling runs once per
    edge. The natural frequencies are drawn from a normal distribution of width `sigma`, and the phases uniformly from
    the circle. Above a critical coupling of the order of the all-to-all value `2 sigma sqrt(2 / pi)`, the order
    parameter `r = |<exp(i theta)>|` grows from about `1 / sqrt(N)` towards 1, and small-world shortcuts lower it.

    Args:
        n_oscillators:
            The number of oscillators, the nodes of the network.

        network:
            The generator, `'watts_strogatz'`, `'erdos_renyi'`, or `'barabasi_albert'`.

        degree:
            The mean degree of the network, even for the Watts-Strogatz ring.

        rewiring:
            The probability of rewiring every edge of the Watts-Strogatz ring.

        K:
            The coupling strength, divided by the mean degree.

        sigma:
            The standard deviation of the natural frequencies.

        seed:
            The seed of the network, the frequencies, and the phases.

    Returns:
        system:
            The model, with the phases `oscillators.theta`, the weights `links.w`, and the order parameter `order`.

    Raises:
        ValueError:
            The network is unknown.

    Examples:
        >>> from skcomplex.models.physics.synchronization import kuramoto_on_networks
        >>> system = kuramoto_on_networks(20, degree=4)
        >>> system.elements[1].check().prepare().size_
        40
    """
    space = _network(network, n_oscillators, degree, rewiring, seed)
    rng = np.random.default_rng(seed + 1)
    mean_degree = 2 * space.graph.number_of_edges() / n_oscillators
    oscillators = LatticeElement(
        'oscillators',
        element=PhaseOscillatorElement,
        space=space,
        values={'theta': rng.uniform(0, 2 * math.pi, n_oscillators), 'omega': rng.normal(0.0, sigma, n_oscillators)},
        visual=VisualSpec('sites', {'states': ['theta'], 'low': 0.0, 'high': 2 * math.pi}),
    )
    links = GroupElement('links', element=LinkElement, space=space, on='edges')
    coupling = EdgeCouplingInteraction('coupling', target='oscillators', links='links', K=K / mean_degree)
    return System(elements=[oscillators, links], interactions=[coupling])


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'map', 'chaos', 'network'),
    references=('Arenas et al. 2008', 'Kaneko 1989'),
    defaults={'until': 100.0, 'dt': 1.0},
    observables=lambda system: [ObservableSpec('spread', group='maps', variable='x', reduce='var')],
)
def coupled_map_lattices_on_networks(
    n_nodes: int = 100,
    *,
    network: str = 'erdos_renyi',
    degree: int = 20,
    rewiring: float = 0.1,
    r: float = 4.0,
    epsilon: float = 0.9,
    seed: int = 0,
) -> System:
    """Return chaotic logistic maps on the nodes of a network, each pulled towards the mean of its neighbours.

    Node `i` follows `x_i(t + 1) = (1 - epsilon) f(x_i) + (epsilon / k_i) sum_j A_ij f(x_j)` with the logistic map
    `f(x) = r x (1 - x)`, a mean over the neighbours of the network. For a coupling `epsilon` large enough against the
    Lyapunov exponent of the map and the spectral gap of the network, the maps synchronize on one chaotic orbit, and
    the variance of the states over the nodes falls to zero. Below it they stay incoherent.

    Args:
        n_nodes:
            The number of maps, the nodes of the network.

        network:
            The generator, `'watts_strogatz'`, `'erdos_renyi'`, or `'barabasi_albert'`.

        degree:
            The mean degree of the network.

        rewiring:
            The probability of rewiring every edge of the Watts-Strogatz ring.

        r:
            The parameter of the logistic map, chaotic at 4.

        epsilon:
            The coupling between 0 and 1.

        seed:
            The seed of the network and the initial states.

    Returns:
        system:
            The model, with the states `maps.x` and their variance over the nodes, `spread`.

    Raises:
        ValueError:
            The network is unknown.

    Examples:
        >>> from skcomplex.models.physics.synchronization import coupled_map_lattices_on_networks
        >>> coupled_map_lattices_on_networks(20).elements[0].check().prepare().size_
        20
    """
    space = _network(network, n_nodes, degree, rewiring, seed)
    rng = np.random.default_rng(seed + 1)
    maps = LatticeElement(
        'maps',
        element=LogisticNodeElement,
        space=space,
        values={'x': rng.uniform(0.0, 1.0, n_nodes), 'r': r, 'epsilon': epsilon},
        visual=VisualSpec('sites', {'states': ['x'], 'low': 0.0, 'high': 1.0}),
    )
    return System(elements=[maps], time=DiscreteTime())
